"""
🃏 扑克牌检测 - CompShare 云 GPU 训练脚本
适用于 CompShare（算力共享）平台
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import time

# ===== 配置 =====
CLASSES = [
    'ht a', 'ht 2', 'ht 3', 'ht 4', 'ht 5', 'ht 6', 'ht 7', 'ht 8', 'ht 9', 'ht 10', 'ht j', 'ht q', 'ht k',
    'hx a', 'hx 2', 'hx 3', 'hx 4', 'hx 5', 'hx 6', 'hx 7', 'hx 8', 'hx 9', 'hx 10', 'hx j', 'hx q', 'hx k',
    'mh a', 'mh 2', 'mh 3', 'mh 4', 'mh 5', 'mh 6', 'mh 7', 'mh 8', 'mh 9', 'mh 10', 'mh j', 'mh q', 'mh k',
    'fk a', 'fk 2', 'fk 3', 'fk 4', 'fk 5', 'fk 6', 'fk 7', 'fk 8', 'fk 9', 'fk 10', 'fk j', 'fk q', 'fk k'
]

def convert_xml_to_yolo(xml_path, output_dir):
    """转换 XML 标注到 YOLO 格式"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)
        
        txt_name = Path(xml_path).stem + '.txt'
        txt_path = os.path.join(output_dir, txt_name)
        
        with open(txt_path, 'w') as f:
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in CLASSES:
                    continue
                class_id = CLASSES.index(name)
                
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                x_center = (xmin + xmax) / 2 / img_w
                y_center = (ymin + ymax) / 2 / img_h
                width = (xmax - xmin) / img_w
                height = (ymax - ymin) / img_h
                
                f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')
        return True
    except Exception as e:
        print(f'⚠️  转换失败 {xml_path}: {e}')
        return False

def prepare_dataset():
    """准备数据集"""
    print('='*60)
    print('📦 准备数据集...')
    print('='*60)
    
    # 创建目录
    os.makedirs('dataset/images/train', exist_ok=True)
    os.makedirs('dataset/images/val', exist_ok=True)
    os.makedirs('dataset/labels/train', exist_ok=True)
    os.makedirs('dataset/labels/val', exist_ok=True)
    
    import shutil
    
    # 转换训练集
    train_xml_dir = 'train'
    if os.path.exists(train_xml_dir):
        xml_files = [f for f in os.listdir(train_xml_dir) if f.endswith('.xml')]
        print(f'   处理训练集: {len(xml_files)} 个文件')
        
        train_count = 0
        for xml_file in xml_files:
            if convert_xml_to_yolo(f'{train_xml_dir}/{xml_file}', 'dataset/labels/train'):
                # 复制对应的图片
                img_name = Path(xml_file).stem
                for ext in ['.jpg', '.JPG', '.jpeg', '.png']:
                    img_path = os.path.join(train_xml_dir, img_name + ext)
                    if os.path.exists(img_path):
                        shutil.copy(img_path, 'dataset/images/train/')
                        train_count += 1
                        break
    
    # 转换测试集
    test_xml_dir = 'test'
    if os.path.exists(test_xml_dir):
        xml_files = [f for f in os.listdir(test_xml_dir) if f.endswith('.xml')]
        print(f'   处理测试集: {len(xml_files)} 个文件')
        
        val_count = 0
        for xml_file in xml_files:
            if convert_xml_to_yolo(f'{test_xml_dir}/{xml_file}', 'dataset/labels/val'):
                # 复制对应的图片
                img_name = Path(xml_file).stem
                for ext in ['.jpg', '.JPG', '.jpeg', '.png']:
                    img_path = os.path.join(test_xml_dir, img_name + ext)
                    if os.path.exists(img_path):
                        shutil.copy(img_path, 'dataset/images/val/')
                        val_count += 1
                        break
    
    print(f'✅ 训练集: {train_count} 张图片')
    print(f'✅ 验证集: {val_count} 张图片')
    
    return train_count, val_count

def create_yaml():
    """创建数据集配置文件"""
    yaml_content = f'''path: {os.path.abspath('dataset')}
train: images/train
val: images/val

names:
'''
    for i, name in enumerate(CLASSES):
        yaml_content += f'  {i}: {name}\n'
    
    with open('poker.yaml', 'w') as f:
        f.write(yaml_content)
    print('✅ 配置文件创建完成')

def train_model():
    """开始训练"""
    from ultralytics import YOLO
    import torch
    
    print('\n' + '='*60)
    print('🚀 开始训练...')
    print('='*60)
    
    # 检测 GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'🎮 GPU: {gpu_name}')
        print(f'   显存: {gpu_memory:.1f} GB')
        device = 0
    else:
        print('⚠️  未检测到 GPU，使用 CPU')
        device = 'cpu'
    
    print(f'\n训练配置:')
    print(f'   Epochs: 50')
    print(f'   Batch: 16')
    print(f'   图片尺寸: 320')
    print(f'   预计时间: 1-1.5 小时')
    
    print(f'\n开始训练...\n')
    
    # 加载模型
    model = YOLO('yolov8n.pt')
    
    # 训练
    start_time = time.time()
    
    results = model.train(
        data='poker.yaml',
        epochs=50,
        imgsz=320,
        batch=16,
        device=device,
        patience=10,
        workers=4,
        project='runs',
        name='poker_gpu',
        verbose=True,
        amp=True,  # 自动混合精度
        cache=True,  # 缓存图片
        save_period=10,  # 每 10 epochs 保存
    )
    
    elapsed = time.time() - start_time
    print(f'\n🎉 训练完成！')
    print(f'   用时: {elapsed/3600:.2f} 小时 ({elapsed/60:.0f} 分钟)')
    
    return results

def export_model():
    """导出 TFLite 模型"""
    from ultralytics import YOLO
    
    print('\n' + '='*60)
    print('📦 导出 TFLite 模型...')
    print('='*60)
    
    best_pt = 'runs/poker_gpu/weights/best.pt'
    if not os.path.exists(best_pt):
        print(f'❌ 模型文件不存在: {best_pt}')
        return
    
    model = YOLO(best_pt)
    
    # 导出 FP16
    print('   导出 FP16...')
    model.export(format='tflite', imgsz=320, half=True)
    
    # 导出 INT8
    print('   导出 INT8...')
    model.export(format='tflite', imgsz=320, int8=True, data='poker.yaml')
    
    # 显示文件信息
    fp16_path = 'runs/poker_gpu/weights/best_float16.tflite'
    int8_path = 'runs/poker_gpu/weights/best_int8.tflite'
    
    print(f'\n✅ 导出完成！')
    
    if os.path.exists(fp16_path):
        fp16_size = os.path.getsize(fp16_path) / 1024 / 1024
        print(f'   FP16: {fp16_size:.2f} MB')
    
    if os.path.exists(int8_path):
        int8_size = os.path.getsize(int8_path) / 1024 / 1024
        print(f'   INT8: {int8_size:.2f} MB')
    
    print(f'\n📂 模型位置: runs/poker_gpu/weights/')
    print(f'   - best.pt (PyTorch)')
    print(f'   - best_float16.tflite (FP16, 推荐)')
    print(f'   - best_int8.tflite (INT8, 更快)')

def show_summary():
    """显示总结"""
    print('\n' + '='*60)
    print('🎉 全部完成！')
    print('='*60)
    print('\n下一步:')
    print('  1. 下载模型文件:')
    print('     runs/poker_gpu/weights/best_float16.tflite')
    print('  2. 关闭 CompShare 实例（避免继续计费）')
    print('  3. 将模型部署到 Android 应用')
    print('\n⚠️  重要: 记得关闭实例！')

if __name__ == '__main__':
    print('='*60)
    print('🃏 扑克牌检测 - CompShare 云 GPU 训练')
    print('='*60)
    
    # 检查依赖
    try:
        import ultralytics
        print(f'✅ ultralytics 版本: {ultralytics.__version__}')
    except ImportError:
        print('❌ 请先安装依赖:')
        print('   pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple')
        exit(1)
    
    try:
        # 1. 准备数据集
        train_count, val_count = prepare_dataset()
        
        if train_count == 0:
            print('❌ 没有找到训练数据！')
            print('   请确保 train/ 和 test/ 目录存在且包含数据')
            exit(1)
        
        # 2. 创建配置
        create_yaml()
        
        # 3. 训练
        train_model()
        
        # 4. 导出
        export_model()
        
        # 5. 总结
        show_summary()
        
    except KeyboardInterrupt:
        print('\n\n⚠️  训练被中断')
        print('   已训练的模型保存在: runs/poker_gpu/weights/last.pt')
        print('   可以使用以下命令继续训练:')
        print('   model = YOLO("runs/poker_gpu/weights/last.pt")')
        print('   model.train(resume=True)')
    except Exception as e:
        print(f'\n❌ 发生错误: {e}')
        import traceback
        traceback.print_exc()
