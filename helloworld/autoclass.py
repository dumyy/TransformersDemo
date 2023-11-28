from transformers import AutoTokenizer,AutoModelForSequenceClassification

if __name__ == '__main__':

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    encode_input = tokenizer("it's a good time.",padding=True, truncation=True, max_length=512, return_tensors='pt')

    '''
    padding 被指定时，如果是bool类型，且为True，此时只有输入的query是个list时才有效，向最长句子对齐，短句子则是进行补全。
    input_id为0， 代表特殊符号[PAD], attention_mask = 0, 代表训练/推理时这些位置被屏蔽。
    padding 被指定为'max_length'时，则是所有句子的长度由max_length决定。统一补全为max_length长度。
    '''

    tokens = tokenizer.convert_ids_to_tokens(encode_input.input_ids[0])
    '''
    方便查看query是怎么被分词的
    '''
    decode_input = tokenizer.decode(encode_input.input_ids[0])

    print(encode_input)
    print(tokens, len(tokens))
    print(decode_input)

    '''
    可以利用tokenizer的相关函数将原始的句子转换成
    '''

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    '''
    如果你使用automodel，模型可能不会指定output layer。
    '''

    print(model)

    result = model(**encode_input)
    print(result.logits, type(result.logits))
    '''
    观察model可以发现，model只有网络的输出logit，并没用进行后处理，因此需要手动添加这一部分。
    '''
    
    from torch.nn.functional import softmax
    from torch import argmax

    prob = softmax(result.logits, dim=-1)
    prob_values = prob.detach().cpu().numpy()

    indice = argmax(prob, dim=-1).detach().cpu().numpy()

    dict_used = model.config.id2label
    
    for index, ind_ in enumerate(indice):
        print(f"{dict_used[ind_]}: {prob_values[index][ind_]}")
        



