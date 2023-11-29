'''
https://huggingface.co/docs/transformers/preprocessing#:~:text=A%20tokenizer%20splits%20text%20into%20tokens%20according%20to%20a%20set%20of%20rules
A tokenizer splits text into tokens according to a set of rules
可以看到tokenizer的主要作用便是，根据规则将文本进行分割。
该说不说，text的前处理要比image的前处理简单很多。
'''

from transformers import AutoTokenizer

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.", 
                              truncation=True, 
                              max_length=128, 
                              padding='max_length',
                              )
    '''
    前处理的一些keys基本都在这里了。
    truncation: 截断标志，根据max_length进行设置
    max_length: text转换为token的最大长度，包括CLS和SEP
    padding: str:max_length时，进行padding，bool:True时，当输入有多个文本时，根据最长文本进行补全。
    '''
    print(encoded_input)
    '''
    encoded_input有3个key: input_ids, attention_mask，有时候会有token_type_ids.表示它属于哪个text。
    '''

    decode_input = tokenizer.decode(encoded_input.input_ids, skip_special_tokens=True)
    '''
    可以使用 skip_special_tokens在解码时，将特殊符号CLS,SEP,PAD去掉。
    '''
    print(decode_input)

    '''
    build tensors:将输入直接转为torch tensor，而不是之前的list
    '''

    query = [
        "But what about second breakfast?",
        "Don't think he knows about second breakfast, Pip.",
        "What about elevensies?"
    ]

    encode_input = tokenizer(query, truncation=True, padding=True, return_tensors='pt')

    print(encode_input)

    decode_input_0 = tokenizer.decode(encode_input.input_ids[0], skip_special_tokens=True)
    print(decode_input_0)

    '''
    我们来看一个ner任务的tokenizer,
    一般会有额外的参数is_split_into_words。为True时，list[]代表一个query的所有数据。
    '''
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")
    ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased")

    split_words = ['his','name','is', 'Tom', 'and', "it's", "in", "new york"]
    combined_words = "his name is Tom and it's in new york"

    encode_input = tokenizer(split_words, is_split_into_words=True, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(encode_input.input_ids[0])


    print(encode_input)
    print(tokens)

    result = ner_model(**encode_input)
    # print(result)

    from torch import softmax, argmax

    print(ner_model.config.id2label)

    soft_p = softmax(result.logits, dim=-1)
    indexes = argmax(soft_p, dim=-1).detach().cpu().numpy()[0]
    print(indexes)
    NER = [ner_model.config.id2label[index] for index in indexes]
    print(NER)


    

    
