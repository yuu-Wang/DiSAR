"""
This code collected some methods from DomainBed (https://github.com/facebookresearch/DomainBed) and other SOTA methods.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, collections, logging
from collections import OrderedDict
from info_nce import InfoNCE
from thop import profile

import utils.misc as misc
from utils.validate import algorithm_validate, algorithm_validate_iw, algorithm_eval_iw, algorithm_eval_mu, algorithm_eval_tsne, algorithm_eval_heat
import modeling.model_manager as models
from modeling.losses import DahLoss
from modeling.nets import LossValley, AveragedModel
from modeling.linear_sae import DRLinearSAE, DRTopKSAEEncoder
from modeling.sae_steering import DomainAwareSAESteering
from modeling.srom_modules import (
    ProjectionHead, FactorMLP, CrossImageDecoder,
    symmetric_infonce, srom_separation_loss
)
from dataset.data_manager import get_post_FundusAug

from backpack import backpack, extend
from backpack.extensions import BatchGrad

ALGORITHMS = [
    'ERM',
    'GDRNet',
    'GDRNetSROM',
    'GREEN',
    'CABNet',
    'MixupNet',
    'MixStyleNet',
    'Fishr',
    'DRGen'
    ]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - validate()
    - save_model()
    - renew_model()
    - predict()
    """
    def __init__(self, num_classes, cfg):
        super(Algorithm, self).__init__()
        self.cfg = cfg
        self.epoch = 0

    def update(self, minibatches):
        raise NotImplementedError
    
    def update_epoch(self, epoch):
        self.epoch = epoch
        return epoch
    
    def validate(self, val_loader, test_loader, writer):
        raise NotImplementedError
    
    def save_model(self, log_path):
        raise NotImplementedError
    
    def renew_model(self, log_path):
        raise NotImplementedError
    
    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """Empirical Risk Minimization (ERM)"""
    def __init__(self, num_classes, cfg):
        super(ERM, self).__init__(num_classes, cfg)

        self.network = models.get_net(cfg)

        # input = torch.randn(16, 3, 224, 224)
        # flops, params = profile(self.network, inputs=(input, ))
        # print('FLOPs = ' + str(flops/1000**3) + 'G')
        # print('Params = ' + str(params/1000**2) + 'M')
        # exit()

        if cfg.BACKBONE == 'swint' or cfg.BACKBONE == 'swint_iw':
            self.classifier = models.get_classifier(1024, cfg)
        else:
            self.classifier = models.get_classifier(self.network.out_features(), cfg)

        self.optimizer = torch.optim.SGD(
            [{"params":self.network.parameters()},
            {"params":self.classifier.parameters()}],
            lr = cfg.LEARNING_RATE,
            momentum = cfg.MOMENTUM,
            weight_decay = cfg.WEIGHT_DECAY,
            nesterov=True)

    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()

        features = self.network(image)
        output = self.classifier(features)
        loss = F.cross_entropy(output, label)

        loss.backward()
        self.optimizer.step()

        return {'loss':loss}

    # def validate(self, val_loader, test_loader, writer):
    #     val_auc = -1
    #     test_auc = -1
    #     if self.epoch <= self.cfg.EPOCHS:
    #         if self.cfg.BACKBONE == 'swint_iw':
    #             val_auc, val_loss = algorithm_validate_iw(self, val_loader, writer, self.epoch, 'val')
    #             test_auc, test_loss = algorithm_validate_iw(self, test_loader, writer, self.epoch, 'test')
    #         else:
    #             val_auc, val_loss = algorithm_validate(self, val_loader, writer, self.epoch, 'val')
    #             test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.epoch, 'test')
    #         if self.epoch == self.cfg.EPOCHS:
    #             self.epoch += 1
    #     else:
    #         if self.cfg.BACKBONE == 'swint_iw':
    #             test_auc, test_loss = algorithm_validate_iw(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
    #         else:
    #             test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
    #         logging.info('Best performance on test domain(s): {}'.format(test_auc))

    #     return val_auc, test_auc
    def validate(self, val_loader, test_loader, writer):
        val_acc = -1
        test_acc = -1
        if self.epoch <= self.cfg.EPOCHS:
            if self.cfg.BACKBONE == 'swint_iw':
                val_acc, val_loss = algorithm_validate_iw(self, val_loader, writer, self.epoch, 'val')
                test_acc, test_loss = algorithm_validate_iw(self, test_loader, writer, self.epoch, 'test')
            else:
                val_acc, val_loss = algorithm_validate(self, val_loader, writer, self.epoch, 'val')
                test_acc, test_loss = algorithm_validate(self, test_loader, writer, self.epoch, 'test')
            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1
        else:
            if self.cfg.BACKBONE == 'swint_iw':
                test_acc, test_loss = algorithm_validate_iw(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
            else:
                test_acc, test_loss = algorithm_validate(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
            logging.info('Last performance on test domain(s): {}'.format(test_acc))

        return val_acc, test_acc


    def save_model(self, log_path):
        logging.info("Saving best model...")
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(log_path, 'best_classifier.pth'))
        # torch.save(self.network.state_dict(), os.path.join(log_path, 'model_e{}_{:.2f}.pth'.format(epoch, test_acc*100)))
        # torch.save(self.classifier.state_dict(), os.path.join(log_path, 'classifier_e{}_{:.2f}.pth'.format(epoch, test_acc*100)))

    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')
        self.network.load_state_dict(torch.load(net_path))
        self.classifier.load_state_dict(torch.load(classifier_path))

    def predict(self, x):
        if self.cfg.BACKBONE == 'swint_iw':
            # feature, iw_loss, sw_loss = self.network(x)
            # return self.classifier(feature), iw_loss, sw_loss
            feature, iw_loss, l4 = self.network(x)
            return self.classifier(feature), iw_loss, l4
        else:
            return self.classifier(self.network(x))

    def load_model(self, model_path):
        net_path = os.path.join(model_path, 'best_model.pth')
        classifier_path = os.path.join(model_path, 'best_classifier.pth')

        print("Loading backbone from:", net_path)
        print("Loading classifier from:", classifier_path)

        self.network.load_state_dict(
            torch.load(net_path, map_location='cuda')
        )

        self.classifier.load_state_dict(
            torch.load(classifier_path, map_location='cuda')
        )

    def eval_cls(self, val_loader, test_loader, writer):
        test_acc = algorithm_eval_iw(
            self,
            test_loader,
            writer,
            self.cfg.EPOCHS + self.cfg.VAL_EPOCH,
            'test'
        )

        logging.info(
            'Best performance on test domain(s): {}'.format(test_acc)
        )

        return test_acc

    def eval_cls(self, val_loader, test_loader, writer):

        test_acc = algorithm_eval_iw(
            self,
            test_loader,
            writer,
            self.cfg.EPOCHS + self.cfg.VAL_EPOCH,
            'test'
        )

        logging.info(
            'Best performance on test domain(s): {}'.format(
                test_acc
            )
        )

        return test_acc

# Our method
class GDRNet(ERM):
    """GDRNet with a TopK Linear SAE between the backbone and classifier."""
    def __init__(self, num_classes, cfg):
        super(GDRNet, self).__init__(num_classes, cfg)
        self.fundusAug = get_post_FundusAug(cfg)
        self.criterion = DahLoss(
            beta=cfg.GDRNET.BETA,
            max_iteration=cfg.EPOCHS,
            training_domains=cfg.DATASET.SOURCE_DOMAINS,
            temperature=cfg.GDRNET.TEMPERATURE,
            scaling_factor=cfg.GDRNET.SCALING_FACTOR,
        )
        self.num_classes = num_classes

        # ERM creates the SGD optimizer before the SAE exists.
        self.sae = DRLinearSAE(
            d_in=self.network.out_features(),
            d_sae=cfg.SAE.LATENT_DIM,
            k=cfg.SAE.TOPK,
        )
        self.optimizer.add_param_group({'params': self.sae.parameters()})

    def img_process(self, img_tensor, mask_tensor, fundusAug):
        img_tensor_new, mask_tensor_new = fundusAug['post_aug1'](
            img_tensor.clone(), mask_tensor.clone())
        img_tensor_new = img_tensor_new * mask_tensor_new
        img_tensor_new = fundusAug['post_aug2'](img_tensor_new)
        img_tensor_ori = fundusAug['post_aug2'](img_tensor)
        return img_tensor_new, img_tensor_ori

    def update(self, minibatch, margin=None):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()
        image_new, image_ori = self.img_process(image, mask, self.fundusAug)

        features_ori = self.network(image_ori)
        features_new = self.network(image_new)

        # Classification remains end-to-end differentiable through Linear SAE.
        cls_features_new, _ = self.sae(features_new)
        output_new = self.classifier(cls_features_new)

        gdr_loss, loss_dict_iter = self.criterion(
            [output_new], [features_ori, features_new], label, domain)

        # Reconstruction learns to match the current features without directly
        # moving the backbone through the reconstruction objective.
        if self.cfg.SAE.RECON_WEIGHT > 0:
            ori_target = features_ori.detach()
            new_target = features_new.detach()
            recon_ori, _ = self.sae(ori_target)
            recon_new, _ = self.sae(new_target)
            recon_loss = 0.5 * (
                F.mse_loss(recon_ori, ori_target) +
                F.mse_loss(recon_new, new_target)
            )
        else:
            recon_loss = features_new.new_zeros(())

        loss = gdr_loss + self.cfg.SAE.RECON_WEIGHT * recon_loss
        loss.backward()
        self.optimizer.step()

        loss_dict_iter['loss'] = loss.item()
        loss_dict_iter['loss_gdr'] = gdr_loss.item()
        loss_dict_iter['loss_recon'] = recon_loss.item()
        return loss_dict_iter

    def update_epoch(self, epoch):
        self.epoch = epoch
        return self.criterion.update_alpha(epoch)

    def predict(self, x):
        features = self.network(x)
        reconstructed, _ = self.sae(features)
        return self.classifier(reconstructed)

    def save_model(self, log_path):
        super().save_model(log_path)
        torch.save(self.sae.state_dict(), os.path.join(log_path, 'best_sae.pth'))

    def renew_model(self, log_path):
        super().renew_model(log_path)
        self.sae.load_state_dict(
            torch.load(os.path.join(log_path, 'best_sae.pth'), map_location='cpu'))

    def load_model(self, model_path):
        self.network.load_state_dict(
            torch.load(os.path.join(model_path, 'best_model.pth'), map_location='cpu'))
        self.classifier.load_state_dict(
            torch.load(os.path.join(model_path, 'best_classifier.pth'), map_location='cpu'))
        self.sae.load_state_dict(
            torch.load(os.path.join(model_path, 'best_sae.pth'), map_location='cpu'))

    def eval_cls(self, val_loader, test_loader, writer):
        test_acc = algorithm_eval_iw(
            self, test_loader, writer,
            self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
        logging.info('Best performance on test domain(s): {}'.format(test_acc))
        return test_acc


class GDRNetSROM(GDRNet):
    """GDRNet + projection + TopK sparse encoder + semantic/variation split + SROM.

    Input pair is (x1=image_ori, x2=image_new) from the existing FundusAug.
    GDRNet DahLoss stays enabled; L_grade is an additional *unweighted* CE.
    GDRNetSROM uses an encoder-only SAE and one shared image-space cross decoder.
    """
    def __init__(self, num_classes, cfg):
        if cfg.BACKBONE != 'resnet50':
            raise ValueError('GDRNetSROM currently requires --backbone resnet50.')
        super().__init__(num_classes, cfg)
        sm = cfg.SROM
        self.projection_head = ProjectionHead(
            self.network.out_features(), sm.PROJ_DIM)
        # Replace the original 2048-d SAE with a projected-feature SAE.
        self.sae = DRTopKSAEEncoder(
            d_in=sm.PROJ_DIM, d_sae=cfg.SAE.LATENT_DIM, k=cfg.SAE.TOPK)
        self.semantic_mlp = FactorMLP(
            cfg.SAE.LATENT_DIM, sm.SEMANTIC_DIM, sm.MLP_HIDDEN)
        self.variation_mlp = FactorMLP(
            cfg.SAE.LATENT_DIM, sm.VARIATION_DIM, sm.MLP_HIDDEN)
        self.variation_align = nn.Linear(sm.VARIATION_DIM, sm.SEMANTIC_DIM)
        self.image_decoder = CrossImageDecoder(sm.SEMANTIC_DIM, sm.VARIATION_DIM)
        self.sae_steering = DomainAwareSAESteering(
            latent_dim=cfg.SAE.LATENT_DIM,
            num_domains=max(1, len(cfg.DATASET.SOURCE_DOMAINS)),
            num_classes=num_classes,
            enabled=sm.STEERING_ENABLED,
            strength=sm.STEERING_STRENGTH,
            warmup_epochs=sm.STEERING_WARMUP_EPOCHS,
            ramp_epochs=sm.STEERING_RAMP_EPOCHS,
            importance_momentum=sm.STEERING_IMPORTANCE_MOMENTUM,
        )
        # Classifier consumes the steered semantic representation.
        self.classifier = models.get_classifier(sm.SEMANTIC_DIM, cfg)
        # Rebuild optimizer: the parent optimizer points to discarded SAE/head.
        self.optimizer = torch.optim.SGD(
            [{'params': module.parameters()} for module in (
                self.network, self.projection_head, self.sae,
                self.semantic_mlp, self.variation_mlp, self.variation_align,
                self.image_decoder, self.classifier, self.sae_steering,
            )],
            lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM,
            weight_decay=cfg.WEIGHT_DECAY, nesterov=True,
        )

    def _represent(self, image):
        h = self.network(image)
        z = self.projection_head(h)
        a = self.sae(z)
        # Keep raw SAE activations for probe attribution, but steer the code
        # before semantic/variation factorization.
        a_steered = self.sae_steering(a)
        s = self.semantic_mlp(a_steered)
        v = self.variation_mlp(a_steered)
        return h, z, a, s, v

    def update(self, minibatch, margin=None):
        image, mask, label, domain = minibatch
        # self.epoch stores the last completed epoch; current training epoch is +1.
        self.sae_steering.set_epoch(self.epoch + 1)
        self.optimizer.zero_grad(set_to_none=True)
        image_new, image_ori = self.img_process(image, mask, self.fundusAug)
        h1, z1, a1, s1, v1 = self._represent(image_ori)
        h2, z2, a2, s2, v2 = self._represent(image_new)

        # Grade prediction uses augmentation branch, as in original GDRNet.
        logits = self.classifier(s2)
        l_gdr, log_dict = self.criterion([logits], [h1, h2], label, domain)
        l_grade = F.cross_entropy(logits, label)

        # The domain probe is trained on detached SAE activations, so its
        # gradients update the probe only, not the backbone/SAE representation.
        probe_activations = torch.cat([a1.detach(), a2.detach()], dim=0)
        probe_domains = torch.cat([domain, domain], dim=0)
        l_domain_probe = self.sae_steering.domain_probe_loss(
            probe_activations, probe_domains
        )

        # Estimate which latents positively support the true-grade margin in
        # the actual classifier (rather than using activation frequency alone).
        # This output-effect signal protects diagnosis-relevant features during
        # subsequent domain-aware steering.
        if self.sae_steering.enabled and a2.requires_grad:
            true_logits = logits.gather(1, label.view(-1, 1)).squeeze(1)
            class_mask = F.one_hot(label, num_classes=logits.shape[-1]).bool()
            other_logits = logits.masked_fill(
                class_mask, torch.finfo(logits.dtype).min
            )
            grade_margin = true_logits - torch.logsumexp(other_logits, dim=1)
            margin_grad = torch.autograd.grad(
                grade_margin.sum(), a2, retain_graph=True, create_graph=False,
                allow_unused=True,
            )[0]
            if margin_grad is not None:
                positive_attr = F.relu(a2.detach() * margin_grad.detach()).mean(dim=0)
                self.sae_steering.update_grade_importance(positive_attr)

        # Source samples in the batch are contrasted as paired views.
        sm = self.cfg.SROM
        l_suff = symmetric_infonce(z1, z2, tau=sm.TAU_SUFF)
        l_min = srom_separation_loss(
            s1, s2, self.variation_align(v1), self.variation_align(v2),
            tau=sm.TAU_MIN,
        )

        if sm.LAMBDA_MAX > 0:
            # Crucial: decoder predicts images, NOT projected SAE features.
            x2_hat = self.image_decoder(s1, v2, image_new.shape[-2:])
            x1_hat = self.image_decoder(s2, v1, image_ori.shape[-2:])
            l_max = 0.5 * (F.mse_loss(x2_hat, image_new.detach()) +
                           F.mse_loss(x1_hat, image_ori.detach()))
        else:
            l_max = logits.new_zeros(())

        loss = (
            l_gdr
            + sm.LAMBDA_GRADE * l_grade
            + sm.LAMBDA_SUFF * l_suff
            + sm.LAMBDA_MIN * l_min
            + sm.LAMBDA_MAX * l_max
        )
        # Main task gradients and domain-probe gradients are backpropagated
        # separately. This prevents the probe loss from directly changing SAE
        # activations and prevents probe gradients from affecting main-model clipping.
        loss.backward()
        if self.sae_steering.enabled and self.sae_steering.num_domains >= 2:
            l_domain_probe.backward()

        if sm.GRAD_CLIP > 0:
            main_modules = (
                self.network, self.projection_head, self.sae,
                self.semantic_mlp, self.variation_mlp, self.variation_align,
                self.image_decoder, self.classifier,
            )
            main_params = [
                p for module in main_modules for p in module.parameters()
                if p.grad is not None
            ]
            if main_params:
                nn.utils.clip_grad_norm_(main_params, max_norm=sm.GRAD_CLIP)
            probe_params = [
                p for p in self.sae_steering.domain_probe.parameters()
                if p.grad is not None
            ]
            if probe_params:
                nn.utils.clip_grad_norm_(probe_params, max_norm=sm.GRAD_CLIP)
        self.optimizer.step()

        log_dict.update({
            'loss': loss.item(), 'loss_gdr': l_gdr.item(),
            'loss_grade': l_grade.item(), 'loss_suff': l_suff.item(),
            'loss_min': l_min.item(),
            'loss_max': l_max.item(),
            'loss_steer_domain_probe': l_domain_probe.item(),
            'active_sae': (a2 > 0).sum(dim=1).float().mean().item(),
        })
        log_dict.update(self.sae_steering.diagnostics())
        return log_dict

    def update_epoch(self, epoch):
        self.epoch = epoch
        self.sae_steering.set_epoch(epoch)
        return self.criterion.update_alpha(epoch)

    def predict(self, x):
        _, _, _, s, _ = self._represent(x)
        return self.classifier(s)

    def save_model(self, log_path):
        super().save_model(log_path)
        for name, module in self._srom_components().items():
            torch.save(module.state_dict(), os.path.join(log_path, f'best_{name}.pth'))

    def renew_model(self, log_path):
        super().renew_model(log_path)
        self._load_srom_components(log_path)

    def load_model(self, model_path):
        super().load_model(model_path)
        self._load_srom_components(model_path)

    def _srom_components(self):
        return {
            'projection_head': self.projection_head,
            'semantic_mlp': self.semantic_mlp,
            'variation_mlp': self.variation_mlp,
            'variation_align': self.variation_align,
            'image_decoder': self.image_decoder,
            'sae_steering': self.sae_steering,
        }

    def _load_srom_components(self, path):
        for name, module in self._srom_components().items():
            module.load_state_dict(torch.load(
                os.path.join(path, f'best_{name}.pth'), map_location='cpu'))


class GREEN(Algorithm):
    def __init__(self, num_classes, cfg):
        super(GREEN, self).__init__(num_classes, cfg)
        self.network = models.get_net(cfg)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr = cfg.LEARNING_RATE,
            momentum = cfg.MOMENTUM,
            weight_decay = cfg.WEIGHT_DECAY,
            nesterov=True)
    
    def update(self, minibatch):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()

        output = self.network(image)
        loss = F.cross_entropy(output, label)
        
        loss.backward()
        self.optimizer.step()
        return {'loss':loss}
    
    def validate(self, val_loader, test_loader, writer):
        val_auc = -1
        test_auc = -1
        if self.epoch <= self.cfg.EPOCHS:
            val_auc, val_loss = algorithm_validate(self, val_loader, writer, self.epoch, 'val')
            test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.epoch, 'test')
            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1
        else:
            test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
            logging.info('Best performance on test domain(s): {}'.format(test_auc))
            
        return val_auc, test_auc
    
    def save_model(self, log_path):
        logging.info("Saving best model...")
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
    
    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        self.network.load_state_dict(torch.load(net_path))
    
    def predict(self, x):
        return self.network(x)
    
class CABNet(ERM):
    
    def __init__(self, num_classes, cfg):
        super(CABNet, self).__init__(num_classes, cfg)
        
class MixStyleNet(ERM):
    
    def __init__(self, num_classes, cfg):
        super(MixStyleNet, self).__init__(num_classes, cfg)
        
class MixupNet(ERM):
    
    def __init__(self, num_classes, cfg):
        super(MixupNet, self).__init__(num_classes, cfg)
        self.criterion_CE = torch.nn.CrossEntropyLoss()
    
    def update(self, minibatch, env_feats=None):
        image, mask, label, domain = minibatch
        self.optimizer.zero_grad()

        inputs, targets_a, targets_b, lam = self.mixup_data(image, label)
        outputs = self.predict(inputs)
        loss = self.mixup_criterion(self.criterion_CE, outputs, targets_a, targets_b, lam)
        
        loss.backward()
        self.optimizer.step()

        return {'loss':loss}
    
    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
class Fishr(ERM):
    def __init__(self, num_classes, cfg):
        super(Fishr, self).__init__(num_classes, cfg)
        
        self.num_groups = cfg.FISHR.NUM_GROUPS

        self.network = models.get_net(cfg)
        self.classifier = extend(
            models.get_classifier(self.network._out_features, cfg)
        )
        self.optimizer = None
        
        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            misc.MovingAverage(cfg.FISHR.EMA, oneminusema_correction=True)
            for _ in range(self.num_groups)
        ]  
        self._init_optimizer()
    
    def _init_optimizer(self):
        self.optimizer = torch.optim.SGD(
            list(self.network.parameters()) + list(self.classifier.parameters()),
            lr = self.cfg.LEARNING_RATE,
            momentum = self.cfg.MOMENTUM,
            weight_decay = self.cfg.WEIGHT_DECAY,
            nesterov=True)
        
    def update(self, minibatch):
        image, mask, label, domain = minibatch
        #self.network.train()

        all_x = image
        all_y = label
        
        len_minibatches = [image.shape[0]]
        
        all_z = self.network(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.cfg.FISHR.PENALTY_ANNEAL_ITERS:
            penalty_weight = self.cfg.FISHR.LAMBDA
            if self.update_count == self.cfg.FISHR.PENALTY_ANNEAL_ITERS != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True)
            #torch.autograd.grad(outputs=loss,inputs=list(self.classifier.parameters()),retain_graph=True, create_graph=True)
            
        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_groups)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_groups):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_groups)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_groups):
            penalty += self.l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_groups

    def l2_between_dicts(self, dict_1, dict_2):
        assert len(dict_1) == len(dict_2)
        dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
        dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
        return (
            torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
            torch.cat(tuple([t.view(-1) for t in dict_2_values]))
        ).pow(2).mean()

# DRGen is built based on Fishr method

class DRGen(Algorithm):
    '''
    Refer to the paper 'DRGen: Domain Generalization in Diabetic Retinopathy Classification' 
    https://link.springer.com/chapter/10.1007/978-3-031-16434-7_61
    
    '''
    def __init__(self, num_classes, cfg):
        super(DRGen, self).__init__(num_classes, cfg)
        algorithm_class = get_algorithm_class('Fishr')
        self.algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)
        self.optimizer = self.algorithm.optimizer
        
        self.swad_algorithm = AveragedModel(self.algorithm)
        self.swad_algorithm.cuda()
        #swad_cls = getattr(swad_module, 'LossValley')
        #swad_cls = LossValley()
        self.swad = LossValley(None, cfg.DRGEN.N_CONVERGENCE, cfg.DRGEN.N_TOLERANCE, cfg.DRGEN.TOLERANCE_RATIO)
        
    def update(self, minibatch):
        loss_dict_iter = self.algorithm.update(minibatch)
        if self.swad:
            self.swad_algorithm.update_parameters(self.algorithm, step = self.epoch)
        return loss_dict_iter
    
    def validate(self, val_loader, test_loader, writer):
        swad_val_auc = -1
        swad_auc = -1
        if self.epoch <= self.cfg.EPOCHS:
            val_auc, val_loss = algorithm_validate(self.algorithm, val_loader, writer, self.epoch, 'val(Fishr)')
            test_auc, test_loss = algorithm_validate(self.algorithm, test_loader, writer, self.epoch, 'test(Fishr)')

            if self.swad:
                def prt_results_fn(results):
                    print(results)

                self.swad.update_and_evaluate(
                    self.swad_algorithm, val_auc, val_loss, prt_results_fn
                )
                
                if self.epoch != self.cfg.EPOCHS:
                    self.swad_algorithm = self.swad.get_final_model()
                    swad_val_auc, swad_val_loss = algorithm_validate(self.swad_algorithm, val_loader, writer, self.epoch, 'val')
                    swad_auc, swad_loss = algorithm_validate(self.swad_algorithm, test_loader, writer, self.epoch, 'test')
                    
                    if hasattr(self.swad, "dead_valley") and self.swad.dead_valley:
                        logging.info("SWAD valley is dead -> not stop !")
                        #break
                    
                    self.swad_algorithm = AveragedModel(self.algorithm)  # reset
            
            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1
                
        else:
            self.swad_algorithm = self.swad.get_final_model()
            logging.warning("Evaluate SWAD ...")
            swad_auc, swad_loss = algorithm_validate(self.swad_algorithm, test_loader, writer, self.cfg.EPOCHS + self.cfg.VAL_EPOCH , 'test')
            logging.info('(last) swad test auc: {}  loss: {}'.format(swad_auc,swad_loss))
            
        return swad_val_auc, swad_auc    
        
    def save_model(self, log_path):
        self.algorithm.save_model(log_path)
    
    def renew_model(self, log_path):
        self.algorithm.renew_model(log_path)
    
    def predict(self, x):
        return self.swad_algorithm.predict(x)