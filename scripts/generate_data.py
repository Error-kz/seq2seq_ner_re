"""
从medical.json生成Seq2Seq训练数据
生成格式: input_text <SEP> output_text
"""
import json
import os
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_seq2seq_data(medical_json_path, output_path):
    """
    从medical.json生成Seq2Seq训练数据
    
    Args:
        medical_json_path: medical.json文件路径
        output_path: 输出训练数据文件路径
    """
    print(f"正在读取数据文件: {medical_json_path}")
    
    with open(medical_json_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    print(f"共读取 {len(data)} 条疾病数据")
    
    train_samples = []
    
    for idx, item in enumerate(data):
        if idx % 100 == 0:
            print(f"处理进度: {idx}/{len(data)}")
        
        disease = item['name']
        
        # 1. 疾病-症状关系
        if 'symptom' in item and item['symptom']:
            symptoms = item['symptom']
            # 过滤空字符串
            symptoms = [s.strip() for s in symptoms if s.strip()]
            if symptoms:
                input_text = f"文本: {disease}有什么症状？ 关系类型: 疾病-症状"
                output_text = "; ".join([f"({disease}, 症状, {s})" for s in symptoms])
                train_samples.append(f"{input_text} <SEP> {output_text}\n")
        
        # 2. 疾病-推荐药品关系
        if 'recommand_drug' in item and item['recommand_drug']:
            drugs = item['recommand_drug']
            drugs = [d.strip() for d in drugs if d.strip()]
            if drugs:
                input_text = f"文本: {disease}推荐什么药品？ 关系类型: 疾病-药品"
                output_text = "; ".join([f"({disease}, 推荐药品, {d})" for d in drugs])
                train_samples.append(f"{input_text} <SEP> {output_text}\n")
        
        # 3. 疾病-并发症关系
        if 'acompany' in item and item['acompany']:
            acompany = item['acompany']
            acompany = [a.strip() for a in acompany if a.strip()]
            if acompany:
                input_text = f"文本: {disease}的并发症有哪些？ 关系类型: 疾病-并发症"
                output_text = "; ".join([f"({disease}, 并发症, {a})" for a in acompany])
                train_samples.append(f"{input_text} <SEP> {output_text}\n")
        
        # 4. 疾病-推荐食物关系
        if 'recommand_eat' in item and item['recommand_eat']:
            foods = item['recommand_eat']
            foods = [f.strip() for f in foods if f.strip()]
            if foods:
                input_text = f"文本: {disease}推荐吃什么食物？ 关系类型: 疾病-食物"
                output_text = "; ".join([f"({disease}, 推荐食物, {f})" for f in foods])
                train_samples.append(f"{input_text} <SEP> {output_text}\n")
        
        # 5. 疾病-忌口食物关系
        if 'not_eat' in item and item['not_eat']:
            not_foods = item['not_eat']
            not_foods = [f.strip() for f in not_foods if f.strip()]
            if not_foods:
                input_text = f"文本: {disease}不能吃什么食物？ 关系类型: 疾病-忌口食物"
                output_text = "; ".join([f"({disease}, 忌口食物, {f})" for f in not_foods])
                train_samples.append(f"{input_text} <SEP> {output_text}\n")
        
        # 6. 疾病-宜吃食物关系
        if 'do_eat' in item and item['do_eat']:
            do_foods = item['do_eat']
            do_foods = [f.strip() for f in do_foods if f.strip()]
            if do_foods:
                input_text = f"文本: {disease}适合吃什么食物？ 关系类型: 疾病-宜吃食物"
                output_text = "; ".join([f"({disease}, 宜吃食物, {f})" for f in do_foods])
                train_samples.append(f"{input_text} <SEP> {output_text}\n")
        
        # 7. 疾病-检查项目关系
        if 'check' in item and item['check']:
            checks = item['check']
            checks = [c.strip() for c in checks if c.strip()]
            if checks:
                input_text = f"文本: {disease}需要做什么检查？ 关系类型: 疾病-检查项目"
                output_text = "; ".join([f"({disease}, 检查项目, {c})" for c in checks])
                train_samples.append(f"{input_text} <SEP> {output_text}\n")
        
        # 8. 疾病-病因关系
        if 'cause' in item and item['cause']:
            cause = item['cause'].strip()
            if cause and len(cause) < 200:  # 限制长度
                input_text = f"文本: {disease}的病因是什么？ 关系类型: 疾病-病因"
                output_text = f"({disease}, 病因, {cause})"
                train_samples.append(f"{input_text} <SEP> {output_text}\n")
        
        # 9. 疾病-预防措施关系
        if 'prevent' in item and item['prevent']:
            prevent = item['prevent'].strip()
            if prevent and len(prevent) < 200:
                input_text = f"文本: {disease}如何预防？ 关系类型: 疾病-预防措施"
                output_text = f"({disease}, 预防措施, {prevent})"
                train_samples.append(f"{input_text} <SEP> {output_text}\n")
        
        # 10. 疾病-治疗方式关系
        if 'cure_way' in item and item['cure_way']:
            cure_ways = item['cure_way']
            if isinstance(cure_ways, list):
                cure_ways = [c.strip() for c in cure_ways if c.strip()]
                if cure_ways:
                    input_text = f"文本: {disease}怎么治疗？ 关系类型: 疾病-治疗方式"
                    output_text = "; ".join([f"({disease}, 治疗方式, {c})" for c in cure_ways])
                    train_samples.append(f"{input_text} <SEP> {output_text}\n")
        
        # 11. 症状-疾病关系（反向查询）
        if 'symptom' in item and item['symptom']:
            symptoms = item['symptom']
            symptoms = [s.strip() for s in symptoms if s.strip()]
            for symptom in symptoms:
                input_text = f"文本: {symptom}可能是什么病？ 关系类型: 症状-疾病"
                output_text = f"({disease}, 症状, {symptom})"
                train_samples.append(f"{input_text} <SEP> {output_text}\n")
        
        # 12. 药品-疾病关系（反向查询）
        if 'recommand_drug' in item and item['recommand_drug']:
            drugs = item['recommand_drug']
            drugs = [d.strip() for d in drugs if d.strip()]
            for drug in drugs:
                input_text = f"文本: {drug}能治什么病？ 关系类型: 药品-疾病"
                output_text = f"({disease}, 推荐药品, {drug})"
                train_samples.append(f"{input_text} <SEP> {output_text}\n")
        
        # 13. 检查项目-疾病关系（反向查询）
        if 'check' in item and item['check']:
            checks = item['check']
            checks = [c.strip() for c in checks if c.strip()]
            for check in checks:
                input_text = f"文本: {check}能查出什么病？ 关系类型: 检查项目-疾病"
                output_text = f"({disease}, 检查项目, {check})"
                train_samples.append(f"{input_text} <SEP> {output_text}\n")
    
    # 保存训练数据
    print(f"\n正在保存训练数据到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(train_samples)
    
    print(f"✅ 成功生成 {len(train_samples)} 条训练样本")
    print(f"数据已保存到: {output_path}")
    
    return len(train_samples)


if __name__ == '__main__':
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # seq2seq_ner_re目录
    seq2seq_dir = os.path.dirname(current_dir)
    # red_spider目录（父目录）
    red_spider_dir = os.path.dirname(seq2seq_dir)
    
    # 设置路径
    medical_json_path = os.path.join(red_spider_dir, 'data', 'medical.json')
    output_path = os.path.join(seq2seq_dir, 'data', 'train_seq2seq.txt')
    
    # 检查输入文件是否存在
    if not os.path.exists(medical_json_path):
        print(f"❌ 错误: 找不到数据文件 {medical_json_path}")
        print(f"   请确认 medical.json 文件是否存在")
        sys.exit(1)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"输入文件: {medical_json_path}")
    print(f"输出文件: {output_path}")
    print()
    
    # 生成数据
    generate_seq2seq_data(medical_json_path, output_path)

