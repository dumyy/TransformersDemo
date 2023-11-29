from transformers import AutoModelForSequenceClassification,AutoModelForMaskedLM,AutoModelForTokenClassification
from transformers import AutoTokenizer

if __name__ == '__main__':

    '''
    以加载distilbert-base-uncased model为例。
    - 已知这是个做掩码预测的model，因此对应的class应该是AutoModelForMaskedLM
    - 尝试使用这个模型做文本分类（text classification）和token分类(token classification)的初始化
    '''
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    '''
    - 首先初始化和权重一致的网络结构
    '''
    masked_model = AutoModelForMaskedLM.from_pretrained(model_name)
    '''
    打印model。可以看到maskedlm的网络结构。
    '''
    print(masked_model)

    query = "it's a [MASK] time."

    input_token = tokenizer(query, truncation=True, return_tensors='pt')
    result = masked_model(**input_token)

    '''
    result是个概率分布，需要将其转换为具体的word。因此需要进行下面两个步骤：
    1. 找到[MASK]的位置
    2. 根据概率分布找到topk索引对应的words。
    下面是一个简易的后处理,实现上不一定很标准
    '''
    # 查找[MASK]特殊标志对应的索引。
    tokens = tokenizer.convert_ids_to_tokens(input_token.input_ids[0])
    print("网络输入：",tokens)
    target_index = tokens.index('[MASK]')
    from torch import softmax,topk
    softmax_prob = softmax(result.logits,dim=-1)
    
    probs, indexes = topk(softmax_prob, k = 5, dim=-1)
    # 我们将索引转换成原始的词

    indexes = indexes.detach().cpu().numpy()[0] #batch size = 1,此处直接去掉维度。

    #简单去除[CLS]和[SEP]的位置
    indexes_ = indexes[1:-1]
    new_seq = tokenizer.decode(indexes_[:,0])
    print("补全句子：",new_seq)

    #当然也可以查找[MASK]的可能词汇
    masked_words = tokenizer.convert_ids_to_tokens(indexes[target_index])
    print("补全top5：", masked_words)

    '''
    上面是一个正常使用的例子，网络的结构和权重对应。
    当面向其他的nlp任务时，同样可以使用这个权重做其他任务的初始化。
    '''

    text_cls_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    token_cls_model = AutoModelForTokenClassification.from_pretrained(model_name)

    '''
    和正常的模型权重初始化一样，它只会初始化能找到的module。
    第第一个text_cls_model为例，它会打印一些警告信息：
    ... at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
    这一部分是分类相关的模块，并不能被loadmodel的权重初始化。
    这一个过程一般用于微调的初始化，可以认为是对backbone加载pretrain weights。
    至于输出部分，以分类为例，和实际的分类任务有关。我们还需要额外指定：num_labels, label2id, id2label等参数。
    下面是一个2分类例子：
    '''
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    text_cls_2_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id)

