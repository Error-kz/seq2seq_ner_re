"""
Seq2Seq模型推理类
用于替换question_classifier和question_parser
"""
import os
import re
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import sys

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import Config


class Seq2SeqNER_RE:
    """Seq2Seq NER+RE模型推理类"""
    
    def __init__(self, model_path=None):
        """
        初始化模型
        
        Args:
            model_path: 模型路径，如果为None则使用config中的默认路径
        """
        config = Config()
        
        if model_path is None:
            model_path = os.path.join(config.MODEL_DIR, 'final_model')
        
        print(f"正在加载模型: {model_path}")
        
        # 加载tokenizer和模型
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()
        
        # 移动到设备
        config = Config()
        self.device = torch.device(config.DEVICE)
        self.model.to(self.device)
        
        self.config = config
        print(f"✅ 模型加载完成，使用设备: {self.device}")
    
    def extract_triplets(self, text, relation_types=None):
        """
        从文本中提取三元组
        
        Args:
            text: 输入文本（用户问题）
            relation_types: 关系类型列表，如果为None则尝试所有关系类型
        
        Returns:
            list: 三元组列表 [{'subject': str, 'relation': str, 'object': str}, ...]
        """
        if relation_types is None:
            relation_types = self.config.RELATION_TYPES
        
        # 构建输入prompt
        relation_str = ", ".join(relation_types)
        input_text = f"文本: {text} 关系类型: {relation_str}"
        
        # Tokenize
        input_ids = self.tokenizer.encode(
            input_text,
            return_tensors='pt',
            max_length=self.config.MAX_LENGTH,
            truncation=True
        ).to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=self.config.MAX_TARGET_LENGTH,
                num_beams=self.config.NUM_BEAMS,
                early_stopping=self.config.EARLY_STOPPING,
                do_sample=self.config.DO_SAMPLE,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # 解码
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 解析三元组
        triplets = self._parse_triplets(output_text)
        return triplets
    
    def _parse_triplets(self, text):
        """
        解析生成的三元组文本
        
        Args:
            text: 模型生成的文本，格式如: (subject, relation, object); (subject, relation, object)
        
        Returns:
            list: 三元组字典列表
        """
        triplets = []
        
        # 匹配格式: (subject, relation, object)
        pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
        matches = re.findall(pattern, text)
        
        for match in matches:
            subject, relation, obj = match
            triplets.append({
                'subject': subject.strip(),
                'relation': relation.strip(),
                'object': obj.strip()
            })
        
        return triplets
    
    def classify(self, question):
        """
        分类函数，兼容原有的question_classifier接口
        
        Args:
            question: 用户问题
        
        Returns:
            dict: 包含'args'和'question_types'的字典，兼容原有格式
        """
        # 提取三元组
        triplets = self.extract_triplets(question)
        
        if not triplets:
            return {}
        
        # 构建返回结果（兼容原有格式）
        result = {
            'args': {},  # 实体字典
            'question_types': []  # 问题类型列表
        }
        
        # 从三元组中提取实体和关系
        seen_types = set()
        
        for triplet in triplets:
            subject = triplet['subject']
            relation = triplet['relation']
            obj = triplet['object']
            
            # 确定实体类型并添加到args
            if subject not in result['args']:
                result['args'][subject] = []
            
            # 根据关系类型确定问题类型
            question_type = self._relation_to_question_type(relation)
            if question_type and question_type not in seen_types:
                result['question_types'].append(question_type)
                seen_types.add(question_type)
            
            # 确定实体类型
            if relation in ['疾病-症状', '疾病-药品', '疾病-食物', '疾病-并发症', 
                           '疾病-忌口食物', '疾病-宜吃食物', '疾病-检查项目', 
                           '疾病-病因', '疾病-预防措施', '疾病-治疗方式']:
                if 'disease' not in result['args'][subject]:
                    result['args'][subject].append('disease')
            elif relation == '症状-疾病':
                if 'symptom' not in result['args'][subject]:
                    result['args'][subject].append('symptom')
            elif relation == '药品-疾病':
                if 'drug' not in result['args'][subject]:
                    result['args'][subject].append('drug')
            elif relation == '检查项目-疾病':
                if 'check' not in result['args'][subject]:
                    result['args'][subject].append('check')
        
        return result
    
    def _relation_to_question_type(self, relation):
        """将关系类型转换为问题类型"""
        mapping = {
            '疾病-症状': 'disease_symptom',
            '症状-疾病': 'symptom_disease',
            '疾病-药品': 'disease_drug',
            '药品-疾病': 'drug_disease',
            '疾病-食物': 'disease_food',
            '疾病-忌口食物': 'disease_not_food',
            '疾病-宜吃食物': 'disease_do_food',
            '疾病-检查项目': 'disease_check',
            '检查项目-疾病': 'check_disease',
            '疾病-并发症': 'disease_acompany',
            '疾病-病因': 'disease_cause',
            '疾病-预防措施': 'disease_prevent',
            '疾病-治疗方式': 'disease_cureway',
        }
        return mapping.get(relation)


class EnhancedQuestionParser:
    """
    增强的问题解析器，使用Seq2Seq模型生成Cypher查询
    兼容原有的question_parser接口
    """
    
    def __init__(self, model_path=None):
        """初始化解析器"""
        self.seq2seq_model = Seq2SeqNER_RE(model_path)
        self.config = Config()
    
    def build_entitydict(self, args):
        """构建实体字典（兼容原有接口）"""
        entity_dict = {}
        for arg, types in args.items():
            for type in types:
                if type not in entity_dict:
                    entity_dict[type] = [arg]
                else:
                    entity_dict[type].append(arg)
        return entity_dict
    
    def parser_main(self, res_classify):
        """
        解析主函数（兼容原有接口）
        
        Args:
            res_classify: classify函数的返回结果
        
        Returns:
            list: Cypher查询列表
        """
        args = res_classify.get('args', {})
        question_types = res_classify.get('question_types', [])
        
        sqls = []
        
        # 从分类结果中提取三元组
        # 这里我们需要重新提取，因为需要生成Cypher
        # 为了简化，我们可以直接使用原有的sql_transfer逻辑
        # 或者使用Seq2Seq模型生成Cypher（更高级）
        
        # 简化版本：使用原有的逻辑
        entity_dict = self.build_entitydict(args)
        
        for question_type in question_types:
            sql_ = {'question_type': question_type}
            sql = []
            
            # 根据问题类型生成Cypher（使用原有逻辑）
            if question_type == 'disease_symptom':
                sql = self._sql_transfer(question_type, entity_dict.get('disease', []))
            elif question_type == 'symptom_disease':
                sql = self._sql_transfer(question_type, entity_dict.get('symptom', []))
            elif question_type == 'disease_drug':
                sql = self._sql_transfer(question_type, entity_dict.get('disease', []))
            # ... 其他类型类似
            
            if sql:
                sql_['sql'] = sql
                sqls.append(sql_)
        
        return sqls
    
    def _sql_transfer(self, question_type, entities):
        """生成Cypher查询（复用原有逻辑）"""
        if not entities:
            return []
        
        sql = []
        # 这里可以复用question_parser.py中的逻辑
        # 或者使用Seq2Seq模型直接生成Cypher（更高级的方法）
        
        # 简化版本：使用模板生成
        if question_type == 'disease_symptom':
            sql = [
                f"MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where m.name = '{e}' return m.name, r.name, n.name"
                for e in entities
            ]
        elif question_type == 'disease_drug':
            sql = [
                f"MATCH (m:Disease)-[r:recommand_drug]->(n:Drug) where m.name = '{e}' return m.name, r.name, n.name"
                for e in entities
            ]
        # ... 其他类型
        
        return sql

