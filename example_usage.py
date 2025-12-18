"""
Seq2Seq模型使用示例
演示如何替换原有的question_classifier和question_parser
"""
import os
import sys

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from models.inference import Seq2SeqNER_RE, EnhancedQuestionParser


def example_basic_usage():
    """基本使用示例"""
    print("=" * 50)
    print("基本使用示例")
    print("=" * 50)
    
    # 初始化模型（需要先训练）
    # model_path = os.path.join(current_dir, 'saved_model', 'final_model')
    # classifier = Seq2SeqNER_RE(model_path)
    
    # 如果没有训练好的模型，先显示如何使用
    print("\n1. 首先需要训练模型:")
    print("   python scripts/train.py")
    
    print("\n2. 然后可以使用模型:")
    print("""
    from models.inference import Seq2SeqNER_RE
    
    classifier = Seq2SeqNER_RE()
    question = "感冒有什么症状？"
    result = classifier.classify(question)
    print(result)
    """)


def example_replace_classifier():
    """替换question_classifier的示例"""
    print("\n" + "=" * 50)
    print("替换question_classifier示例")
    print("=" * 50)
    
    code = '''
# 在question_classifier.py中替换

# 原有代码:
# class QuestionClassifier:
#     def __init__(self):
#         # 基于规则的方法
#         ...

# 新代码:
from seq2seq_ner_re.models.inference import Seq2SeqNER_RE

class QuestionClassifier:
    def __init__(self):
        # 使用Seq2Seq模型
        self.model = Seq2SeqNER_RE()
    
    def classify(self, question):
        return self.model.classify(question)
'''
    print(code)


def example_hybrid_approach():
    """混合方法示例（推荐）"""
    print("\n" + "=" * 50)
    print("混合方法示例（推荐）")
    print("=" * 50)
    
    code = '''
# 保留原有规则方法作为后备，Seq2Seq作为主要方法

from question_classifier import QuestionClassifier as RuleClassifier
from seq2seq_ner_re.models.inference import Seq2SeqNER_RE

class HybridQuestionClassifier:
    def __init__(self):
        self.rule_classifier = RuleClassifier()  # 原有方法
        self.seq2seq_model = Seq2SeqNER_RE()    # 新方法
    
    def classify(self, question):
        # 优先使用Seq2Seq模型
        try:
            result = self.seq2seq_model.classify(question)
            if result and result.get('question_types'):
                return result
        except Exception as e:
            print(f"Seq2Seq模型失败: {e}")
        
        # 如果失败，使用规则方法
        return self.rule_classifier.classify(question)
'''
    print(code)


def example_test_triplets():
    """测试三元组提取"""
    print("\n" + "=" * 50)
    print("测试三元组提取")
    print("=" * 50)
    
    code = '''
from models.inference import Seq2SeqNER_RE

classifier = Seq2SeqNER_RE()

# 测试问题
questions = [
    "感冒有什么症状？",
    "百日咳推荐什么药品？",
    "发烧可能是什么病？",
]

for question in questions:
    triplets = classifier.extract_triplets(question)
    print(f"问题: {question}")
    print(f"三元组: {triplets}")
    print()
'''
    print(code)


if __name__ == '__main__':
    example_basic_usage()
    example_replace_classifier()
    example_hybrid_approach()
    example_test_triplets()
    
    print("\n" + "=" * 50)
    print("使用说明")
    print("=" * 50)
    print("""
使用步骤:
1. 生成训练数据:
   cd seq2seq_ner_re/scripts
   python generate_data.py

2. 训练模型:
   python scripts/train.py

3. 使用模型:
   参考上面的示例代码
    """)

