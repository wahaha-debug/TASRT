import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.backbones.t2t import t2t_vit_t_24
from fvcore.nn import flop_count
from utils.flops import give_supported_ops
import copy
from modeling.meta_arch import build_transformer, weights_init_classifier, weights_init_kaiming
import torch
from modeling.clip import clip
from modeling.fusion_part.CDA_Module import CDA
from utils.simple_tokenizer import SimpleTokenizer


class IDEA(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(IDEA, self).__init__()
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = self.feat_dim
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 512
        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
        self.num_classes = num_classes
        self.cfg = cfg
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        self.camera = camera_num
        self.view = view_num
        self.direct = cfg.MODEL.DIRECT
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.image_size = cfg.INPUT.SIZE_TRAIN
        self.miss_type = cfg.TEST.MISS
        self.DA = cfg.MODEL.DA
        self.q_size = cfg.INPUT.SIZE_TRAIN[0] // 16, cfg.INPUT.SIZE_TRAIN[1] // 16
        self.window_size = self.q_size
        self.stride_block = self.q_size
        if self.DA:
            self.CDA = CDA(q_size=self.q_size, window_size=self.q_size, ksize=4,
                                                               stride=2,
                                                               stride_block=self.q_size,
                                                               offset_range_factor=cfg.MODEL.OFF_FAC,
                                                               share=cfg.MODEL.DA_SHARE)
            self.num_region = self.CDA.num_da
            self.visual_classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.visual_classifier.apply(weights_init_classifier)
            self.bottleneck_visual = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck_visual.bias.requires_grad_(False)
            self.bottleneck_visual.apply(weights_init_kaiming)

            self.textual_classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.textual_classifier.apply(weights_init_classifier)
            self.bottleneck_textual = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck_textual.bias.requires_grad_(False)
            self.bottleneck_textual.apply(weights_init_kaiming)

            print('~~~~~~~~~~~~~~~Using CDA~~~~~~~~~~~~~~~')
        if self.direct:
            self.classifier_v = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier_v.apply(weights_init_classifier)
            self.bottleneck_v = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck_v.bias.requires_grad_(False)
            self.bottleneck_v.apply(weights_init_kaiming)

            self.classifier_t = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier_t.apply(weights_init_classifier)
            self.bottleneck_t = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck_t.bias.requires_grad_(False)
            self.bottleneck_t.apply(weights_init_kaiming)
        else:
            self.classifier_t_rgb = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_t_rgb.apply(weights_init_classifier)
            self.bottleneck_t_rgb = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_t_rgb.bias.requires_grad_(False)
            self.bottleneck_t_rgb.apply(weights_init_kaiming)

            self.classifier_t_nir = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_t_nir.apply(weights_init_classifier)
            self.bottleneck_t_nir = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_t_nir.bias.requires_grad_(False)
            self.bottleneck_t_nir.apply(weights_init_kaiming)

            self.classifier_t_tir = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_t_tir.apply(weights_init_classifier)
            self.bottleneck_t_tir = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_t_tir.bias.requires_grad_(False)
            self.bottleneck_t_tir.apply(weights_init_kaiming)

            self.classifier_v_nir = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_v_nir.apply(weights_init_classifier)
            self.bottleneck_v_nir = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_v_nir.bias.requires_grad_(False)
            self.bottleneck_v_nir.apply(weights_init_kaiming)

            self.classifier_v_tir = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_v_tir.apply(weights_init_classifier)
            self.bottleneck_v_tir = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_v_tir.bias.requires_grad_(False)
            self.bottleneck_v_tir.apply(weights_init_kaiming)

            self.classifier_v_rgb = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_v_rgb.apply(weights_init_classifier)
            self.bottleneck_v_rgb = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_v_rgb.bias.requires_grad_(False)
            self.bottleneck_v_rgb.apply(weights_init_kaiming)

        self.tokenizer = SimpleTokenizer()

    def load_param(self, trained_path):
        state_dict = torch.load(trained_path, map_location="cpu")
        print(f"Successfully load ckpt!")
        incompatibleKeys = self.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)

    def flops(self, shape=(3, 256, 128)):
        if self.image_size[0] != shape[1] or self.image_size[1] != shape[2]:
            shape = (3, self.image_size[0], self.image_size[1])
            # For vehicle reid, the input shape is (3, 128, 256)
        supported_ops = give_supported_ops()
        model = copy.deepcopy(self)
        model.cuda().eval()
        input_r = torch.ones((1, *shape), device=next(model.parameters()).device, dtype=torch.float32)
        input_n = torch.ones((1, *shape), device=next(model.parameters()).device, dtype=torch.float32)
        input_t = torch.ones((1, *shape), device=next(model.parameters()).device, dtype=torch.float32)
        cam_label = torch.tensor(0, device=next(model.parameters()).device, dtype=torch.int64)
        input = {"RGB": input_r, "NI": input_n, "TI": input_t, "cam_label": cam_label,
                 'text': {'rgb_text': clip.tokenize('just a test').cuda(),
                          'ni_text': clip.tokenize('just a test').cuda(),
                          'ti_text': clip.tokenize('just a test').cuda()}}
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(
            "The out_proj here is called by the nn.MultiheadAttention, which has been calculated in th .forward(), so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("For the bottleneck or classifier, it is not calculated during inference, so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(
            "For the Mamba Series, the code implementations are all used with the inner weight instead of directly calling the model, the FLOPs has been calculated with our inner function 'MambaInnerFn_jit', so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        del model, input
        return sum(Gflops.values()) * 1e9

    def forward(self, image, text=None, label=None, cam_label=None, view_label=None, return_pattern=3, img_path=None,
                writer=None, epoch=None):
        if 'cam_label' in image:
            cam_label = image['cam_label']
        if 'text' in image:
            RGB_Text = image['text']['rgb_text']
            NI_Text = image['text']['ni_text']
            TI_Text = image['text']['ti_text']
        else:
            RGB_Text = text['rgb_text']
            NI_Text = text['ni_text']
            TI_Text = text['ti_text']
        real_text_rgb = []
        real_text_nir = []
        real_text_tir = []
        for i in range(len(RGB_Text)):
            real_text_rgb.append(self.tokenizer.decode(RGB_Text[i].tolist()))
            real_text_nir.append(self.tokenizer.decode(NI_Text[i].tolist()))
            real_text_tir.append(self.tokenizer.decode(TI_Text[i].tolist()))
        text_real = {'rgb_text': real_text_rgb, 'ni_text': real_text_nir, 'ti_text': real_text_tir}
        if self.training:
            RGB = image['RGB']
            NI = image['NI']
            TI = image['TI']
            RGB_v_feas, RGB_v_global, RGB_t_feas, RGB_t_global = self.BACKBONE(image=RGB, text=RGB_Text,
                                                                               cam_label=cam_label, label=label,
                                                                               view_label=view_label)
            NI_v_feas, NI_v_global, NI_t_feas, NI_t_global = self.BACKBONE(image=NI, text=NI_Text, cam_label=cam_label,
                                                                           label=label,
                                                                           view_label=view_label)
            TI_v_feas, TI_v_global, TI_t_feas, TI_t_global = self.BACKBONE(image=TI, text=TI_Text, cam_label=cam_label,
                                                                           label=label,
                                                                           view_label=view_label)
            if self.DA:
                boss_fea = torch.stack([RGB_v_global, NI_v_global, TI_v_global, RGB_t_global, NI_t_global, TI_t_global],
                                       dim=1)
                visual, textual = self.CDA(RGB_v_feas, NI_v_feas, TI_v_feas, boss_fea, writer=writer,
                                                             epoch=epoch,
                                                             img_path=img_path)
                score_vv = self.visual_classifier(self.bottleneck_visual(visual))
                score_tt = self.textual_classifier(self.bottleneck_textual(textual))
            if self.direct:
                ori_v = torch.cat([RGB_v_global, NI_v_global, TI_v_global], dim=-1)
                ori_t = torch.cat([RGB_t_global, NI_t_global, TI_t_global], dim=-1)
                score_v = self.classifier_v(self.bottleneck_v(ori_v))
                score_t = self.classifier_t(self.bottleneck_t(ori_t))
                if self.DA:
                    return score_v, ori_v, score_t, ori_t, score_vv, visual, score_tt, textual
                else:
                    return score_v, ori_v, score_t, ori_t
            else:
                score_rgb_t = self.classifier_t_rgb(self.bottleneck_t_rgb(RGB_t_global))
                score_nir_t = self.classifier_t_nir(self.bottleneck_t_nir(NI_t_global))
                score_tir_t = self.classifier_t_tir(self.bottleneck_t_tir(TI_t_global))
                score_rgb_v = self.classifier_v_rgb(self.bottleneck_v_rgb(RGB_v_global))
                score_nir_v = self.classifier_v_nir(self.bottleneck_v_nir(NI_v_global))
                score_tir_v = self.classifier_v_tir(self.bottleneck_v_tir(TI_v_global))
                if self.DA:
                    return score_rgb_v, RGB_v_global, score_nir_v, NI_v_global, score_tir_v, TI_v_global, \
                        score_rgb_t, RGB_t_global, score_nir_t, NI_t_global, score_tir_t, TI_t_global, score_vv, visual, score_tt, textual
                else:
                    return score_rgb_v, RGB_v_global, score_nir_v, NI_v_global, score_tir_v, TI_v_global, \
                        score_rgb_t, RGB_t_global, score_nir_t, NI_t_global, score_tir_t, TI_t_global

        else:
            RGB = image['RGB']
            NI = image['NI']
            TI = image['TI']
            if self.miss_type == 'r':
                RGB = torch.zeros_like(RGB)
            elif self.miss_type == 'n':
                NI = torch.zeros_like(NI)
            elif self.miss_type == 't':
                TI = torch.zeros_like(TI)
            elif self.miss_type == 'rn':
                RGB = torch.zeros_like(RGB)
                NI = torch.zeros_like(NI)
            elif self.miss_type == 'rt':
                RGB = torch.zeros_like(RGB)
                TI = torch.zeros_like(TI)
            elif self.miss_type == 'nt':
                NI = torch.zeros_like(NI)
                TI = torch.zeros_like(TI)

            NI_v_feas, NI_v_global, NI_t_feas, NI_t_global = self.BACKBONE(image=NI, text=NI_Text, cam_label=cam_label,
                                                                           view_label=view_label)
            RGB_v_feas, RGB_v_global, RGB_t_feas, RGB_t_global = self.BACKBONE(image=RGB, text=RGB_Text,
                                                                               cam_label=cam_label,
                                                                               view_label=view_label)
            TI_v_feas, TI_v_global, TI_t_feas, TI_t_global = self.BACKBONE(image=TI, text=TI_Text, cam_label=cam_label,
                                                                           view_label=view_label)
            multi_modal_dict = {"V_RGB": RGB_v_global, "V_NIR": NI_v_global, "V_TIR": TI_v_global,
                                "T_RGB": RGB_t_global, "T_NIR": NI_t_global, "T_TIR": TI_t_global,
                                'LOCAL_v': torch.cat([RGB_v_global, NI_v_global, TI_v_global], dim=-1),
                                'LOCAL_t': torch.cat([RGB_t_global, NI_t_global, TI_t_global], dim=-1),
                                'LOCAL': torch.cat(
                                    [RGB_v_global, NI_v_global, TI_v_global, RGB_t_global, NI_t_global, TI_t_global],
                                    dim=-1)}
            if self.DA:
                boss_fea = torch.stack([RGB_v_global, NI_v_global, TI_v_global, RGB_t_global, NI_t_global, TI_t_global],
                                       dim=1)
                visual, textual = self.CDA(RGB_v_feas, NI_v_feas, TI_v_feas, boss_fea, writer=writer,
                                                             epoch=epoch,
                                                             img_path=img_path, texts=text_real)
                local = torch.cat([visual, textual], dim=-1)
                multi_modal_dict['LOCAL_v'] = visual
                multi_modal_dict['LOCAL_t'] = textual
                multi_modal_dict['LOCAL'] = local
            return multi_modal_dict


class IDEA_woText(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(IDEA_woText, self).__init__()
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = self.feat_dim
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 512
        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
        self.num_classes = num_classes
        self.cfg = cfg
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        self.camera = camera_num
        self.view = view_num
        self.direct = cfg.MODEL.DIRECT
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.image_size = cfg.INPUT.SIZE_TRAIN
        self.miss_type = cfg.TEST.MISS
        self.DA = cfg.MODEL.DA
        self.q_size = cfg.INPUT.SIZE_TRAIN[0] // 16, cfg.INPUT.SIZE_TRAIN[1] // 16
        self.window_size = self.q_size
        self.stride_block = self.q_size
        if self.DA:
            self.CDA = CDA(q_size=self.q_size, window_size=self.q_size, ksize=4,
                                                               stride=2,
                                                               stride_block=self.q_size,
                                                               offset_range_factor=cfg.MODEL.OFF_FAC,
                                                               share=cfg.MODEL.DA_SHARE)
            self.num_region = self.CDA.num_da
            self.visual_classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.visual_classifier.apply(weights_init_classifier)
            self.bottleneck_visual = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck_visual.bias.requires_grad_(False)
            self.bottleneck_visual.apply(weights_init_kaiming)

            print('~~~~~~~~~~~~~~~Using CDA~~~~~~~~~~~~~~~')
        if self.direct:
            self.classifier_v = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier_v.apply(weights_init_classifier)
            self.bottleneck_v = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck_v.bias.requires_grad_(False)
            self.bottleneck_v.apply(weights_init_kaiming)

        else:
            self.classifier_v_nir = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_v_nir.apply(weights_init_classifier)
            self.bottleneck_v_nir = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_v_nir.bias.requires_grad_(False)
            self.bottleneck_v_nir.apply(weights_init_kaiming)

            self.classifier_v_tir = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_v_tir.apply(weights_init_classifier)
            self.bottleneck_v_tir = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_v_tir.bias.requires_grad_(False)
            self.bottleneck_v_tir.apply(weights_init_kaiming)

            self.classifier_v_rgb = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_v_rgb.apply(weights_init_classifier)
            self.bottleneck_v_rgb = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_v_rgb.bias.requires_grad_(False)
            self.bottleneck_v_rgb.apply(weights_init_kaiming)

    def load_param(self, trained_path):
        state_dict = torch.load(trained_path, map_location="cpu")
        print(f"Successfully load ckpt!")
        incompatibleKeys = self.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)

    def flops(self, shape=(3, 256, 128)):
        if self.image_size[0] != shape[1] or self.image_size[1] != shape[2]:
            shape = (3, self.image_size[0], self.image_size[1])
            # For vehicle reid, the input shape is (3, 128, 256)
        supported_ops = give_supported_ops()
        model = copy.deepcopy(self)
        model.cuda().eval()
        input_r = torch.ones((1, *shape), device=next(model.parameters()).device, dtype=torch.float32)
        input_n = torch.ones((1, *shape), device=next(model.parameters()).device, dtype=torch.float32)
        input_t = torch.ones((1, *shape), device=next(model.parameters()).device, dtype=torch.float32)
        cam_label = torch.tensor(0, device=next(model.parameters()).device, dtype=torch.int64)
        input = {"RGB": input_r, "NI": input_n, "TI": input_t, "cam_label": cam_label,
                 'text': {'rgb_text': clip.tokenize('just a test').cuda(),
                          'ni_text': clip.tokenize('just a test').cuda(),
                          'ti_text': clip.tokenize('just a test').cuda()}}
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(
            "The out_proj here is called by the nn.MultiheadAttention, which has been calculated in th .forward(), so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("For the bottleneck or classifier, it is not calculated during inference, so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(
            "For the Mamba Series, the code implementations are all used with the inner weight instead of directly calling the model, the FLOPs has been calculated with our inner function 'MambaInnerFn_jit', so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        del model, input
        return sum(Gflops.values()) * 1e9


    def forward(self, image, text=None, label=None, cam_label=None, view_label=None, return_pattern=3, img_path=None,
                writer=None, epoch=None):
        if 'cam_label' in image:
            cam_label = image['cam_label']
        RGB_Text = None
        NI_Text = None
        TI_Text = None
        if self.training:
            RGB = image['RGB']
            NI = image['NI']
            TI = image['TI']
            RGB_v_feas, RGB_v_global, RGB_t_feas, RGB_t_global = self.BACKBONE(image=RGB, text=RGB_Text,
                                                                               cam_label=cam_label, label=label,
                                                                               view_label=view_label)
            NI_v_feas, NI_v_global, NI_t_feas, NI_t_global = self.BACKBONE(image=NI, text=NI_Text, cam_label=cam_label,
                                                                           label=label,
                                                                           view_label=view_label)
            TI_v_feas, TI_v_global, TI_t_feas, TI_t_global = self.BACKBONE(image=TI, text=TI_Text, cam_label=cam_label,
                                                                           label=label,
                                                                           view_label=view_label)
            # loss = 0
            if self.DA:
                boss_fea = torch.stack([RGB_v_global, NI_v_global, TI_v_global], dim=1)
                visual, textual = self.CDA(RGB_v_feas, NI_v_feas, TI_v_feas, boss_fea, writer=writer,
                                                             epoch=epoch,
                                                             img_path=img_path)
                score_vv = self.visual_classifier(self.bottleneck_visual(visual))
            if self.direct:
                ori_v = torch.cat([RGB_v_global, NI_v_global, TI_v_global], dim=-1)
                score_v = self.classifier_v(self.bottleneck_v(ori_v))
                if self.DA:
                    return score_v, ori_v, score_vv, visual
                else:
                    return score_v, ori_v
            else:
                score_rgb_v = self.classifier_v_rgb(self.bottleneck_v_rgb(RGB_v_global))
                score_nir_v = self.classifier_v_nir(self.bottleneck_v_nir(NI_v_global))
                score_tir_v = self.classifier_v_tir(self.bottleneck_v_tir(TI_v_global))
                if self.DA:
                    return score_rgb_v, RGB_v_global, score_nir_v, NI_v_global, score_tir_v, TI_v_global, score_vv, visual
                else:
                    return score_rgb_v, RGB_v_global, score_nir_v, NI_v_global, score_tir_v, TI_v_global

        else:
            RGB = image['RGB']
            NI = image['NI']
            TI = image['TI']
            if self.miss_type == 'r':
                RGB = torch.zeros_like(RGB)
            elif self.miss_type == 'n':
                NI = torch.zeros_like(NI)
            elif self.miss_type == 't':
                TI = torch.zeros_like(TI)
            elif self.miss_type == 'rn':
                RGB = torch.zeros_like(RGB)
                NI = torch.zeros_like(NI)
            elif self.miss_type == 'rt':
                RGB = torch.zeros_like(RGB)
                TI = torch.zeros_like(TI)
            elif self.miss_type == 'nt':
                NI = torch.zeros_like(NI)
                TI = torch.zeros_like(TI)

            RGB_v_feas, RGB_v_global, RGB_t_feas, RGB_t_global = self.BACKBONE(image=RGB, text=RGB_Text,
                                                                               cam_label=cam_label,
                                                                               view_label=view_label)
            NI_v_feas, NI_v_global, NI_t_feas, NI_t_global = self.BACKBONE(image=NI, text=NI_Text, cam_label=cam_label,
                                                                           view_label=view_label)
            TI_v_feas, TI_v_global, TI_t_feas, TI_t_global = self.BACKBONE(image=TI, text=TI_Text, cam_label=cam_label,
                                                                           view_label=view_label)

            multi_modal_dict = {"V_RGB": RGB_v_global, "V_NIR": NI_v_global, "V_TIR": TI_v_global,
                                "T_RGB": RGB_t_global, "T_NIR": NI_t_global, "T_TIR": TI_t_global,
                                'LOCAL_v': torch.cat([RGB_v_global, NI_v_global, TI_v_global], dim=-1),
                                'LOCAL_t': torch.cat([RGB_t_global, NI_t_global, TI_t_global], dim=-1),
                                'LOCAL': torch.cat(
                                    [RGB_v_global, NI_v_global, TI_v_global, RGB_t_global, NI_t_global, TI_t_global],
                                    dim=-1)}
            if self.DA:
                boss_fea = torch.stack([RGB_v_global, NI_v_global, TI_v_global], dim=1)
                visual, textual = self.CDA(RGB_v_feas, NI_v_feas, TI_v_feas, boss_fea, writer=writer,
                                                             epoch=epoch,
                                                             img_path=img_path)
                local = torch.cat([visual, textual], dim=-1)
                multi_modal_dict['LOCAL_v'] = visual
                multi_modal_dict['LOCAL_t'] = textual
                multi_modal_dict['LOCAL'] = local
            return multi_modal_dict


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_24': t2t_vit_t_24,
}


def make_model(cfg, num_class, camera_num, view_num=0):
    model = IDEA(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building IDEA===========')
    return model
