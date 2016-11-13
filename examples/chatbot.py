# coding:utf-8
'''
source code for train chatbot with DNC
'''
from DNC.DNC import DNC, optimizers, xp, Variable, cuda, np, _gpu_id, onehot
import chainer
from sklearn.externals import joblib


class ChatBot():
    def __init__(self):
        self.X = 5
        self.Y = 5
        self.N = 10
        self.W = 10
        self.R = 3

        # self.mdl = DNC(self.X, self.Y, self.N, self.W, self.R)
        # self.opt = optimizers.Adam()
        # self.opt.setup(self.mdl)
        self.mdl = None
        self.opt = None
        self.id2wordDict = {}
        self.word2idDict = {}
        self.loss = 0
        self.acc = 0

    def predict(self,x_list):
        datacnt = 1
        content_str_list = x_list
        content = map(lambda i: self.word2idDict[i], content_str_list)

        seqlen = len(content)

        x_seq_list = [float('nan')] * seqlen
        for i in range(seqlen):
            x_seq_list[i] = onehot(content[i], self.X)

        self.mdl.reset_state()
        y_list = []
        for cnt in range(seqlen):
            x = Variable(cuda.to_gpu(x_seq_list[cnt].reshape(1, self.X), _gpu_id))
            y = self.mdl(x)
            y_list.append(self.id2wordDict[int(y.data.argmax())])
        return y_list

    def train(self, x_list, y_list):
        datanum = len(x_list)

        word_set = set()
        for each in x_list:
            word_set.update(each)
        for each in y_list:
            word_set.update(each)
        self.X = len(word_set)
        self.Y = len(word_set)
        self.mdl = DNC(self.X, self.Y, self.N, self.W, self.R)
        self.opt = optimizers.Adam()
        self.opt.setup(self.mdl)
        loss = 0.0
        acc = 0.0
        self.id2wordDict = dict(zip(range(len(word_set)), list(word_set)))
        self.word2idDict = dict(map(lambda (k, v): (v, k), self.id2wordDict.items()))
        for batch in range(1,101):
            print "batch", batch
            if batch%5 == 0:
                print "check!!!", batch
                for test_list in x_list:
                    print "test question:", " ".join(test_list)
                    answer_list = chatbot.predict(test_list)
                    print "asnwer:", " ".join(answer_list)

            for datacnt in range(datanum):
                lossfrac = xp.zeros((1, 2))
                content_str_list = input_list[datacnt]
                content = map(lambda i: self.word2idDict[i], content_str_list)

                true_content_str_list = output_list[datacnt]
                true_content = map(lambda i: self.word2idDict[i], true_content_str_list)

                contentlen = len(content)
                seqlen = len(content)
                x_seq_list = [float('nan')] * seqlen
                t_seq_list = [float('nan')] * seqlen
                for i in range(seqlen):
                    x_seq_list[i] = onehot(content[i], self.X)
                    t_seq_list[i] = onehot(true_content[i], self.X)

                self.mdl.reset_state()
                for cnt in range(seqlen):
                    x = Variable(cuda.to_gpu(x_seq_list[cnt].reshape(1, self.X), _gpu_id))
                    if (isinstance(t_seq_list[cnt], xp.ndarray)):
                        t = Variable(cuda.to_gpu(t_seq_list[cnt].reshape(1, self.Y), _gpu_id))
                    else:
                        t = []
                    y = self.mdl(x)
                    if (isinstance(t, chainer.Variable)):
                        loss += (y - t) ** 2
                        if (np.argmax(y.data) == np.argmax(t.data)): acc += 1
                    if (cnt == seqlen - 1):
                        self.mdl.cleargrads()
                        print batch, '(', datacnt, ')', loss.data[0].sum(), acc

                        loss.grad = xp.ones(loss.data.shape, dtype=np.float32)
                        loss.backward()
                        self.opt.update()
                        loss.unchain_backward()
                        print batch, '(', datacnt, ')', loss.data.sum() / loss.data.size / contentlen, acc / contentlen
                        lossfrac += xp.array([loss.data.sum() / loss.data.size / seqlen, 1.], np.float32)
                        loss = 0.0
                        acc = 0.0




if __name__ == '__main__':
    input_list = [["<start>", u"오늘", u"날씨가", u"어때", "<eos>"], ["<start>", u"오늘", u"온도는", u"어때", "<eos>"], \
                  ["<start>", u"내일", u"비가", u"올까", "<eos>"]]
    output_list = [[u"화창하고", u"맑고", u"쾌적하고", u"좋습니다", "<eos>"], [u"오늘", u"온도는", u"무척", u"높습니다", "<eos>"], \
                   [u"내일은", u"비가", u"옵니다", "<eos>", ""]]
    chatbot = ChatBot()
    chatbot.train(input_list,output_list)
    joblib.dump(chatbot,"/home/dev/data/DNC_test.4.model")
    for test_list in input_list[::-1]:
        print "test question:",  " ".join(test_list)
        answer_list = chatbot.predict(test_list)
        print "asnwer:"," ".join(answer_list)
