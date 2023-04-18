# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss



class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7

        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            #print('pi shape ',pi.shape)
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets

        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        #print('177 targets shape',targets.shape)
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            # print('targets ,',targets[:,:,2:6])
            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches

                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                # print(r)
                # print(1/r)
                # print('r shape ',r.shape)
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # print(j)
                # print('j shape ',j)
                t = t[j]  # filter
                # print('t',t.shape)
                # Offsets
                gxy = t[:, 2:4]  # grid xy
                # print('gxy ,',gxy)
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                # print('k',k)
                # print('totoal',((gxy % 1 < g) & (gxy > 1)))
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

class ComputeLoss_co:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7

        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p1, p2,targets,remerber_rate=0.8):  # predictions, targets, model
        device = targets.device
        lobj1 = torch.zeros(1, device=device)  ###loss obj of p1
        lobj2 = torch.zeros(1, device=device)  ###loss obj of p2
        tcls1, tbox1, indices1, anchors1 = self.build_targets1(p1, targets)  # targets
        tcls2, tbox2, indices2, anchors2 = self.build_targets1(p2, targets) # targets
        # Losses
        lbox_1 = []
        lcls_1 = []
        lbox_2 = []
        lcls_2 = []
        for i, pi in enumerate(p1):  # layer index, layer predictions p1
            #print('pi shape ',pi.shape)
            b, a, gj, gi = indices1[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors1[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox1[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)

                lbox_temp = (1.0 - iou)
                lbox_1.append(lbox_temp) #loss of box
                #lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls1[i]] = self.cp
                    self.BCEcls.reduction='none'
                    lcls_temp = self.BCEcls(ps[:, 5:], t).mean(1)
                    #print(t.shape)
                    lcls_1.append(lcls_temp)
                    #self.BCEcls.reduction='mean'
                    #lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj1 += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        for i, pi in enumerate(p2):  # layer index, layer predictions  p2
            #print('pi shape ',pi.shape)
            b, a, gj, gi = indices2[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors1[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox2[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)

                lbox_temp = (1.0 - iou)
                lbox_2.append(lbox_temp)  # loss of box
                #lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls2[i]] = self.cp
                    self.BCEcls.reduction='none'
                    lcls_temp = self.BCEcls(ps[:, 5:], t).mean(1)
                    lcls_2.append(lcls_temp)
                    #self.BCEcls.reduction='mean'
                    #lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj2 += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
         # lcls_1=[]
        if lbox_1==[]:
            lbox_1=torch.tensor([0],dtype=torch.float32, device=device)
            lbox_1 = torch.cat(lbox_1, 0)
        else:
            lbox_1 = torch.cat(lbox_1,0)
        if lcls_1==[]:
            lcls_1=torch.tensor([0],dtype=torch.float32, device=device)
            lcls_1 = torch.cat(lcls_1, 0)
        else:
            lcls_1 = torch.cat(lcls_1,0)
        l1 = lbox_1+lcls_1
        # print(" l1",l1.shape)
        sort_id1 = torch.argsort(l1)
        num = int(remerber_rate*l1.shape[0])
        # print(l1)
        # print(sort_id1)
        # print('lbox shape',lbox_1.shape)
        # print('lcls shape', lcls_1.shape)
        if lbox_2==[]:
            lbox_2=torch.tensor([0],dtype=torch.float32,device=device)
            lbox_2 = torch.cat(lbox_2, 0)
        else:
            lbox_2 = torch.cat(lbox_2, 0)
        if lcls_2==[]:
            lcls_2 = torch.tensor([0],dtype=torch.float32,device=device)
            lcls_2 = torch.cat(lcls_2, 0)
        else:
            lcls_2 = torch.cat(lcls_2, 0)
        l2 = lbox_2+lcls_2
        # print('lbox shape', lbox_1.shape)
        # print('lcls shape', lcls_1.shape)

        sort_id2 = torch.argsort(l2)
        lbox_1_update = lbox_1[sort_id2[:num]]
        lcls_1_update = lcls_1[sort_id2[:num]]
        lbox_2_update = lbox_2[sort_id1[:num]]
        lcls_2_update = lcls_2[sort_id1[:num]]

        lbox1 = lbox_1_update.mean(0,keepdim=True)
        lcls1 = lcls_1_update.mean(0,keepdim=True)

        lbox2 = lbox_2_update.mean(0,keepdim=True)
        lcls2 = lcls_2_update.mean(0,keepdim=True)

        lbox1 *= self.hyp['box']
        lobj1 *= self.hyp['obj']
        lcls1 *= self.hyp['cls']

        lbox2 *= self.hyp['box']
        lobj2 *= self.hyp['obj']
        lcls2 *= self.hyp['cls']
        # print(lobj2)
        # print(lobj2.shape)
        bs = tobj.shape[0]  # batch size

        return (lbox1 + lobj1 + lcls1) * bs, torch.cat((lbox1, lobj1, lcls1)).detach() ,(lbox2 + lobj2 + lcls2) * bs, torch.cat((lbox2, lobj2, lcls2)).detach()

    def build_targets1(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets

        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        #print('177 targets shape',targets.shape)
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            # print('targets ,',targets[:,:,2:6])
            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches

                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                # print(r)
                # print(1/r)
                # print('r shape ',r.shape)
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # print(j)
                # print('j shape ',j)
                t = t[j]  # filter
                # print('t',t.shape)
                # Offsets
                gxy = t[:, 2:4]  # grid xy
                # print('gxy ,',gxy)
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                # print('k',k)
                # print('totoal',((gxy % 1 < g) & (gxy > 1)))
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

class ComputeLoss_co_imgs:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7

        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p1, p2,targets,remerber_rate=0.8):  # predictions, targets, model
        device = targets.device
        loss1, loss1_items = self.compute_loss_imgs(p1, targets)
        # sort_id1 = torch.argsort(torch.tensor(loss1))
        # print(sort_id1)
        sort_id1 = [index for index, value in sorted(list(enumerate(loss1)), key=lambda x: x[1])]
        # print(sort_id1)
        loss1_sorted = [loss1[i] for i in sort_id1]
        # print(loss1_items_sorted)

        loss2, loss2_items =self.compute_loss_imgs(p2, targets)
        sort_id2 = [index for index, value in sorted(list(enumerate(loss2)), key=lambda x: x[1])]
        # print(sort_id1)
        loss2_sorted = [loss2[i] for i in sort_id2]
        # print(loss1_sorted)
        ###coteaching without reweight
        num_remember = int(remerber_rate * len(loss1_sorted))
        id1_update = sort_id1[:num_remember]
        id2_update = sort_id2[:num_remember]

        loss1_update = [loss1[i] for i in id2_update]
        loss1_items_update = [loss1_items[i] for i in id2_update]
        loss1_items_update_batch = 0
        for _id, loss_items in enumerate(loss1_items_update):
            loss1_items_update_batch += loss_items
        loss1_items_update_batch /= num_remember
        loss1_update_batch = 0
        for _id, loss1 in enumerate(loss1_update):
            loss1_update_batch += loss1
        loss1_update_batch /= num_remember

        loss2_update = [loss2[i] for i in id1_update]
        loss2_items_update = [loss2_items[i] for i in id1_update]
        loss2_items_update_batch = 0
        for _id, loss_items in enumerate(loss2_items_update):
            loss2_items_update_batch += loss_items
        loss2_items_update_batch /= num_remember
        loss2_update_batch = 0
        for _id, loss2 in enumerate(loss2_update):
            loss2_update_batch += loss2
        loss2_update_batch /= num_remember

        # print('loss1',loss1_sorted)
        # print(sum(loss1_sorted))
        # print('loss1 update',loss1_update_batch)
        # print('loss2 ',torch.sum(loss2)/num_remember)
        # print('loss2 update ',loss2_update_batch)
        ###coteaching without reweight
        # num_remember = int(remember_rate * len(loss1_sorted))
        # id1_update = sort_id1[:num_remember]
        # id2_update = sort_id2[:num_remember]
        #
        ####coteaching with reweight
        return loss1_update_batch, loss1_items_update_batch.detach(), loss2_update_batch, loss2_items_update_batch.detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets

        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        #print('177 targets shape',targets.shape)
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            # print('targets ,',targets[:,:,2:6])
            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches

                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                # print(r)
                # print(1/r)
                # print('r shape ',r.shape)
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # print(j)
                # print('j shape ',j)
                t = t[j]  # filter
                # print('t',t.shape)
                # Offsets
                gxy = t[:, 2:4]  # grid xy
                # print('gxy ,',gxy)
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                # print('k',k)
                # print('totoal',((gxy % 1 < g) & (gxy > 1)))
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def compute_loss(self,p, targets):  # predictions, targets, model

        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        h =self.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['cls_pw']], device=device))  # weight=model.class_weights)
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # Losses
        nt = 0  # number of targets
        no = len(p)  # number of outputs
        balance = [4.0, 1.0, 0.3, 0.1, 0.03]  # P3-P7
        for i, pi in enumerate(p):  # layer index, layer predictions

            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                # tobj[b, a, gj, gi] = 1.0 # iou ratio
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                    t[range(n), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

        s = 3 / no  # output count scaling
        lbox *= h['box'] * s
        lobj *= h['obj']
        lcls *= h['cls'] * s
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        # print('loss compute_loss',loss)
        return loss * bs, torch.cat((lbox, lobj, lcls)).detach()

    def compute_loss_imgs(self,p, targets):  #
        p_for_imgs = []
        targets_for_imgs = []
        batch = p[0].shape[0]
        for _id in range(batch):
            p_for_layers = []
            for _i, pi in enumerate(p):
                p_for_layers.append(torch.unsqueeze(pi[_id, ...], 0))
            p_for_imgs.append(p_for_layers)
        for _id in range(batch):
            index = torch.where(targets[:, 0] == _id)
            target = torch.index_select(targets, 0, index[0])
            target[:, 0] = 0
            targets_for_imgs.append(target)
            # print(index)
        loss = []
        loss_items = []
        for _id in range(batch):
            # print('batch id', _id)
            # print('targets ,', targets_for_imgs[_id])
            # print('pi ,', p_for_imgs[_id][0].shape)
            loss_temp, loss_item_temp = self.compute_loss(p_for_imgs[_id], targets_for_imgs[_id])
            loss.append(loss_temp)
            loss_items.append(loss_item_temp)
        # print(len(loss))
        # print(loss_items)
        # print('loss ', loss)
        # loss = torch.tensor(loss)
        return loss, loss_items

class ComputeLoss_co_re:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7

        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p1, p2,targets,remerber_rate=0.8):  # predictions, targets, model
        device = targets.device
        lobj1 = torch.zeros(1, device=device)  ###loss obj of p1
        lobj2 = torch.zeros(1, device=device)  ###loss obj of p2
        tcls1, tbox1, indices1, anchors1 = self.build_targets1(p1, targets)  # targets
        tcls2, tbox2, indices2, anchors2 = self.build_targets1(p2, targets) # targets
        # Losses
        lbox_1 = []
        lcls_1 = []
        lbox_2 = []
        lcls_2 = []
        for i, pi in enumerate(p1):  # layer index, layer predictions p1
            #print('pi shape ',pi.shape)
            b, a, gj, gi = indices1[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors1[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox1[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)

                lbox_temp = (1.0 - iou)
                lbox_1.append(lbox_temp) #loss of box
                #lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls1[i]] = self.cp
                    self.BCEcls.reduction='none'
                    lcls_temp = self.BCEcls(ps[:, 5:], t).mean(1)
                    #print(t.shape)
                    lcls_1.append(lcls_temp)
                    #self.BCEcls.reduction='mean'
                    #lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj1 += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        for i, pi in enumerate(p2):  # layer index, layer predictions  p2
            #print('pi shape ',pi.shape)
            b, a, gj, gi = indices2[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors1[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox2[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)

                lbox_temp = (1.0 - iou)
                lbox_2.append(lbox_temp)  # loss of box
                #lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls2[i]] = self.cp
                    self.BCEcls.reduction='none'
                    lcls_temp = self.BCEcls(ps[:, 5:], t).mean(1)
                    lcls_2.append(lcls_temp)
                    #self.BCEcls.reduction='mean'
                    #lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj2 += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        if lbox_1==[]:
            lbox_1=torch.tensor([0])
        else:
            lbox_1 = torch.cat(lbox_1,0)
        if lcls_1==[]:
            lcls_1=torch.tensor([0])
        else:
            lcls_1 = torch.cat(lcls_1,0)
        l1 = lbox_1+lcls_1
        # print(l1)
        sort_id1 = torch.argsort(l1)
        num = int(remerber_rate*l1.shape[0])
        # print(l1)
        # print(sort_id1)
        # print('lbox shape',lbox_1.shape)
        # print('lcls shape', lcls_1.shape)
        # print('lobj shape', lobj1.shape)
        if lbox_2==[]:
            lbox_2=torch.tensor([0])
        else:
            lbox_2 = torch.cat(lbox_2, 0)
        if lcls_2==[]:
            lcls_2 = torch.tensor([0])
        else:
            lcls_2 = torch.cat(lcls_2, 0)
        l2 = lbox_2+lcls_2
        # print('lbox shape', lbox_1.shape)
        # print('lcls shape', lcls_1.shape)
        sort_id2 = torch.argsort(l2)
        lbox_1_update = torch.cat([lbox_1[sort_id2[:num]],0.2*lbox_1[sort_id2[num:]]],0)
        lcls_1_update = torch.cat([lcls_1[sort_id2[:num]],0.2*lcls_1[sort_id2[num:]]],0)
        lbox_2_update = torch.cat([lbox_2[sort_id1[:num]],0.2*lbox_2[sort_id1[num:]]],0)
        lcls_2_update = torch.cat([lcls_2[sort_id1[:num]],0.2*lcls_2[sort_id1[num:]]],0)

        lbox1 = lbox_1_update.mean(0,keepdim=True)
        lcls1 = lcls_1_update.mean(0,keepdim=True)

        lbox2 = lbox_2_update.mean(0,keepdim=True)
        lcls2 = lcls_2_update.mean(0,keepdim=True)

        lbox1 *= self.hyp['box']
        lobj1 *= self.hyp['obj']
        lcls1 *= self.hyp['cls']

        lbox2 *= self.hyp['box']
        lobj2 *= self.hyp['obj']
        lcls2 *= self.hyp['cls']
        # print(lobj2)
        # print(lobj2.shape)
        bs = tobj.shape[0]  # batch size

        return (lbox1 + lobj1 + lcls1) * bs, torch.cat((lbox1, lobj1, lcls1)).detach() ,(lbox2 + lobj2 + lcls2) * bs, torch.cat((lbox2, lobj2, lcls2)).detach()

    def build_targets1(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets

        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        #print('177 targets shape',targets.shape)
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            # print('targets ,',targets[:,:,2:6])
            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches

                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                # print(r)
                # print(1/r)
                # print('r shape ',r.shape)
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # print(j)
                # print('j shape ',j)
                t = t[j]  # filter
                # print('t',t.shape)
                # Offsets
                gxy = t[:, 2:4]  # grid xy
                # print('gxy ,',gxy)
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                # print('k',k)
                # print('totoal',((gxy % 1 < g) & (gxy > 1)))
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
