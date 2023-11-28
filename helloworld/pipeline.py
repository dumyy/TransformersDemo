from transformers import pipeline


def BertClassifierEN():
    classifer = pipeline(task="sentiment-analysis")
    '''
    我们打印出模型结构，可以发现它是一个蒸馏版本的bert模型，transformer部分只有6个block，
    transformer attention部分hidden_dim = 768，但是查不到有几个head，这部分需要check 预训练模型的config.json文件。
    bert的输出层接了一个简单的2分类。
    '''
    print(classifer.model)
    '''
    打印模型的config，可以看到模型的详细信息，如模型名，attention和ffn的各个维度，可以发现multihead是12，6个layer。
    '''
    print(classifer.model.config)

    result = classifer("it's a good time.")
    print(result)

    result = classifer("it's too hard for me.")
    print(result)

def BertClassifierZH():
    classifier = pipeline(task="sentiment-analysis", model="hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2")
    print(classifier.model.config)
    print(classifier.model)
    
    result = classifier("今天天气真好啊，我们去郊游吧！")
    print(result)


if __name__ == '__main__':

    BertClassifierEN()

    BertClassifierZH()
