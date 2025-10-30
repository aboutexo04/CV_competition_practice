# src/model.py

"""
모델 정의 및 생성 모듈
"""
import torch
import torch.nn as nn
import timm

# 모델별 설정
MODEL_CONFIGS = {
    'efficientnet_b0': {'display_name': 'EfficientNet-B0', 'short_name': 'effnet-b0'},
    'efficientnet_b1': {'display_name': 'EfficientNet-B1', 'short_name': 'effnet-b1'},
    'efficientnet_b2': {'display_name': 'EfficientNet-B2', 'short_name': 'effnet-b2'},
    'efficientnet_b3': {'display_name': 'EfficientNet-B3', 'short_name': 'effnet-b3'},
    'resnet18': {'display_name': 'ResNet-18', 'short_name': 'resnet18'},
    'resnet34': {'display_name': 'ResNet-34', 'short_name': 'resnet34'},
    'resnet50': {'display_name': 'ResNet-50', 'short_name': 'resnet50'},
    'resnet101': {'display_name': 'ResNet-101', 'short_name': 'resnet101'},
    'vit_tiny_patch16_224': {'display_name': 'ViT-Tiny', 'short_name': 'vit-tiny'},
    'vit_small_patch16_224': {'display_name': 'ViT-Small', 'short_name': 'vit-small'},
    'convnext_tiny': {'display_name': 'ConvNeXt-Tiny', 'short_name': 'convnext-tiny'},
    'convnext_small': {'display_name': 'ConvNeXt-Small', 'short_name': 'convnext-small'},
    'mobilenetv3_small_100': {'display_name': 'MobileNetV3-Small', 'short_name': 'mobilenet-small'},
    'mobilenetv3_large_100': {'display_name': 'MobileNetV3-Large', 'short_name': 'mobilenet-large'},
}


def get_model(model_name, num_classes=10, pretrained=True):
    """
    Config 기반으로 모델 생성
    
    Args:
        model_name: 모델 이름 (예: 'efficientnet_b0', 'resnet50')
        num_classes: 출력 클래스 수
        pretrained: ImageNet pretrained weights 사용 여부
        
    Returns:
        model: PyTorch 모델
    """
    try:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # Display name 가져오기
        model_config = MODEL_CONFIGS.get(
            model_name, 
            {'display_name': model_name, 'short_name': model_name}
        )
        display_name = model_config['display_name']
        
        print(f"✅ 모델 생성 완료: {display_name} ({model_name})")
        if pretrained:
            print(f"   Pretrained weights 사용")
        
        return model
    
    except Exception as e:
        print(f"❌ 모델 생성 실패: {model_name}")
        print(f"   Error: {e}")
        raise


def print_model_list(current_model=None):
    """
    사용 가능한 모든 모델 목록을 카테고리별로 출력
    
    Args:
        current_model: 현재 선택된 모델 이름 (강조 표시용)
    """
    print("=" * 70)
    print("📋 사용 가능한 모델 목록")
    print("=" * 70)

    # 모델을 카테고리별로 그룹화
    model_groups = {
        'EfficientNet': [k for k in MODEL_CONFIGS.keys() if 'efficientnet' in k],
        'ResNet': [k for k in MODEL_CONFIGS.keys() if 'resnet' in k],
        'ViT (Vision Transformer)': [k for k in MODEL_CONFIGS.keys() if 'vit' in k],
        'ConvNeXt': [k for k in MODEL_CONFIGS.keys() if 'convnext' in k],
        'MobileNet': [k for k in MODEL_CONFIGS.keys() if 'mobilenet' in k],
    }

    idx = 1
    for group_name, models in model_groups.items():
        if models:
            print(f"\n🔹 {group_name}:")
            for model_name in models:
                config = MODEL_CONFIGS[model_name]
                marker = "👉 " if model_name == current_model else "   "
                print(f"{marker}{idx:2d}. {config['display_name']:25s} ({model_name})")
                idx += 1

    print("\n" + "=" * 70)
    if current_model:
        model_config = MODEL_CONFIGS.get(
            current_model,
            {'display_name': current_model}
        )
        print(f"✅ 현재 선택된 모델: {model_config['display_name']} ({current_model})")
    print(f"💡 모델 변경 방법: config.update(MODEL_NAME='모델명')")
    print("=" * 70)


def get_model_info(model):
    """
    모델 정보 반환

    Args:
        model: PyTorch 모델

    Returns:
        dict: 모델 정보 딕셔너리
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params
    }


def print_model_info(model, device, model_name=None):
    """
    모델 정보를 깔끔하게 출력

    Args:
        model: PyTorch 모델
        device: 모델이 위치한 디바이스
        model_name: 모델 이름 (선택)
    """
    info = get_model_info(model)
    
    # Display name 가져오기
    if model_name:
        model_config = MODEL_CONFIGS.get(
            model_name,
            {'display_name': model_name}
        )
        display_name = model_config['display_name']
    else:
        display_name = "Unknown"

    print(f"\n📊 모델 정보")
    print("=" * 70)
    print(f"  모델 이름:           {display_name}")
    print(f"  전체 파라미터:       {info['total_params']:,}")
    print(f"  학습 가능 파라미터:  {info['trainable_params']:,}")
    print(f"  고정 파라미터:       {info['frozen_params']:,}")
    print(f"  디바이스:            {device}")
    print("=" * 70)


def get_optimizer(model, config):
    """
    Config 기반으로 optimizer 생성
    
    Args:
        model: PyTorch 모델
        config: Config 객체
        
    Returns:
        optimizer: PyTorch optimizer
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LR,
        weight_decay=getattr(config, 'WEIGHT_DECAY', 1e-4)
    )
    
    return optimizer


def get_scheduler(optimizer, config):
    """
    Config 기반으로 learning rate scheduler 생성
    
    Args:
        optimizer: PyTorch optimizer
        config: Config 객체
        
    Returns:
        scheduler: PyTorch scheduler
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=1e-6
    )
    
    return scheduler


def freeze_layers(model, freeze_ratio=0.5):
    """
    모델의 일부 레이어를 동결

    Args:
        model: PyTorch 모델
        freeze_ratio (float): 동결할 레이어 비율 (0.0 ~ 1.0)

    Returns:
        model: 레이어가 동결된 모델
    """
    params = list(model.parameters())
    freeze_count = int(len(params) * freeze_ratio)

    for i, param in enumerate(params):
        if i < freeze_count:
            param.requires_grad = False

    return model


def unfreeze_all(model):
    """
    모델의 모든 레이어를 학습 가능하게 설정

    Args:
        model: PyTorch 모델

    Returns:
        model: 모든 레이어가 학습 가능한 모델
    """
    for param in model.parameters():
        param.requires_grad = True

    return model


# ============================================
# Backward compatibility
# ============================================

def create_model(model_name, num_classes=10, pretrained=True):
    """기존 함수명 (backward compatibility)"""
    return get_model(model_name, num_classes, pretrained)