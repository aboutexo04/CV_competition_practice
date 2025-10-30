# ============================================
# CIFAR10 클래스 분포 분석
# ============================================
from collections import Counter

# CIFAR10 클래스 이름 로드 (batches.meta에서 직접 로드)
# 항상 데이터에서 직접 로드하여 정확한 클래스명 사용
import pickle
meta_path = train_data.data_dir / 'batches.meta'
with open(meta_path, 'rb') as f:
    meta = pickle.load(f, encoding='bytes')
if b'label_names' in meta:
    class_names = [name.decode('utf-8') for name in meta[b'label_names']]
elif 'label_names' in meta:
    class_names = meta['label_names']
else:
    raise ValueError("Could not find label_names in batches.meta file")

# Train 클래스 카운트 (CIFAR10Dataset은 .labels 속성을 가지고 있음)
train_labels = train_data.labels.tolist()
train_counts = Counter(train_labels)

# Test 클래스 카운트
test_labels = test_data.labels.tolist()
test_counts = Counter(test_labels)

# 통계
print("=" * 60)
print("CIFAR10 Train Class Distribution")
print("=" * 60)
for i, name in enumerate(class_names):
    count = train_counts.get(i, 0)
    pct = count / len(train_data) * 100
    print(f"{i}. {name:20s}: {count:6,d} ({pct:5.2f}%)")

print(f"\nTotal train samples: {len(train_data):,}")

print("\n" + "=" * 60)
print("CIFAR10 Test Class Distribution")
print("=" * 60)
for i, name in enumerate(class_names):
    count = test_counts.get(i, 0)
    pct = count / len(test_data) * 100
    print(f"{i}. {name:20s}: {count:6,d} ({pct:5.2f}%)")

print(f"\nTotal test samples: {len(test_data):,}")

# ============================================
# Cell 6: 클래스 분포 시각화
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Train 분포
axes[0].bar(range(len(class_names)), 
            [train_counts[i] for i in range(len(class_names))],
            color='steelblue',
            edgecolor='black',
            alpha=0.7)
axes[0].set_xticks(range(len(class_names)))
axes[0].set_xticklabels(class_names, rotation=45, ha='right')
axes[0].set_title('Train Class Distribution', 
                  fontsize=14, fontweight='bold', pad=20)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

# Test 분포
axes[1].bar(range(len(class_names)), 
            [test_counts[i] for i in range(len(class_names))],
            color='coral',
            edgecolor='black',
            alpha=0.7)
axes[1].set_xticks(range(len(class_names)))
axes[1].set_xticklabels(class_names, rotation=45, ha='right')
axes[1].set_title('Test Class Distribution', 
                  fontsize=14, fontweight='bold', pad=20)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()

# 출력 디렉토리 생성 (없으면 생성)
import os
output_dir = '../outputs'
os.makedirs(output_dir, exist_ok=True)

# 이미지 저장
plt.savefig(os.path.join(output_dir, 'class_distribution.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

# 불균형 체크
max_count = max(train_counts.values())
min_count = min(train_counts.values())
imbalance_ratio = max_count / min_count

print(f"\n{'='*60}")
print(f"⚖️  Class Imbalance Ratio: {imbalance_ratio:.2f}")
if imbalance_ratio > 2.0:
    print("⚠️  Warning: Imbalanced dataset!")
    print("💡 Consider: Class weighting or resampling")
else:
    print("✅ Balanced dataset")
print(f"{'='*60}")

# ============================================
# 클래스별 샘플 이미지
# ============================================

fig, axes = plt.subplots(10, 5, figsize=(15, 25))
fig.suptitle('Sample Images per Class (5 samples each)', 
             fontsize=16, fontweight='bold', y=0.998)

for class_idx in range(10):
    # 해당 클래스 이미지 찾기
    class_indices = [i for i, (_, label) in enumerate(train_data) 
                    if label == class_idx]
    
    # 5개 샘플
    for sample_idx in range(5):
        img, label = train_data[class_indices[sample_idx]]
        
        ax = axes[class_idx, sample_idx]
        ax.imshow(img)
        ax.axis('off')
        
        # 첫 번째 열에 클래스 이름
        if sample_idx == 0:
            ax.text(-0.1, 0.5, class_names[class_idx],
                   transform=ax.transAxes,
                   fontsize=10,
                   fontweight='bold',
                   va='center',
                   ha='right',
                   rotation=0)

plt.tight_layout()
plt.savefig('../outputs/sample_images.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print("✅ Sample images saved!")

# ============================================
# 이미지 크기 & 품질
# ============================================

# 샘플 100개 (CIFAR10Dataset은 numpy array 반환)
sample_images = [train_data[i][0] for i in range(100)]

# 크기 분석 (numpy array는 shape 속성 사용)
sizes = [img.shape for img in sample_images]
widths = [s[1] for s in sizes]  # (H, W, C) 형태이므로 height가 첫번째, width가 두번째
heights = [s[0] for s in sizes]

# 밝기 분석 (이미 numpy array이므로 변환 불필요)
brightnesses = []
contrasts = []

for img in sample_images:
    # CIFAR10Dataset은 이미 numpy array (32, 32, 3) 반환
    brightness = np.mean(img)
    contrast = np.std(img)
    
    brightnesses.append(brightness)
    contrasts.append(contrast)

# 출력
print(f"{'='*60}")
print("📏 Image Dimensions:")
print(f"{'='*60}")
print(f"Width:  {min(widths)} ~ {max(widths)} "
      f"(avg: {np.mean(widths):.1f})")
print(f"Height: {min(heights)} ~ {max(heights)} "
      f"(avg: {np.mean(heights):.1f})")

print(f"\n{'='*60}")
print("💡 Image Quality:")
print(f"{'='*60}")
print(f"Brightness: {np.mean(brightnesses):.1f} ± "
      f"{np.std(brightnesses):.1f}")
print(f"Contrast:   {np.mean(contrasts):.1f} ± "
      f"{np.std(contrasts):.1f}")
print(f"{'='*60}")