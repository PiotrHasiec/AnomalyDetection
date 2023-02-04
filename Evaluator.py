from sklearn.metrics import f1_score,precision_score,recall_score
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
class Evaluator():
    def __init__(self,test_data,anomally_data,preproces_error = None,model = None):
        # self.model = model
        self.test_data = test_data
        self.anomally_data = anomally_data
        self.preproces_error = preproces_error
        self.errors=None
        self.y_visible = None
        self.y_all = None

    def p_r_f1(self,y_true,y_pred):
        gruped_true = [y_true[0]]
        gruped_pred = [y_pred[0]]
        try:
            for i in range(1,len(y_true)-1,50):
                if y_pred[i]!=gruped_pred[-1] or y_true[i]!=gruped_true[-1]:
                    gruped_true.extend(y_true[i:i+49])
                    gruped_pred.extend(y_pred[i:i+49])
        except Exception as i:
            print(i.args)
            print(i)
        return f1_score(gruped_true,gruped_pred),precision_score(gruped_true,gruped_pred),recall_score(gruped_true,gruped_pred)

    # def find_treshold(self,model,path2fig = None,show = False):
    #     test_x,test_y = self.test_data

    #     try:           
    #         predicted_data = model.predict(test_x,verbose=0)
    #     except Exception as inst:
    #                     print(type(inst))    # the exception instance
    #                     print(inst.args)     # arguments stored in .args
    #                     print(inst) 

    #     predicted_data = predicted_data.reshape((-1,test_y.shape[-1]))
    #     all_errors = (test_x.reshape((-1,test_y.shape[-1])) - predicted_data)**2
    #     errors = [np.sqrt(np.sum(err)) for err in all_errors]
        
                
    #     if show:
    #         if self.preproces_error is None:
    #             counts, bins = np.histogram(self.preproces_error(errors),30)
    #         else:
    #             counts, bins = np.histogram(errors,30)
    #         plt.stairs(counts, bins)
    #         plt.show()
    #         if not path2fig is None:
    #             plt.savefig(path2fig)
    #     elif not path2fig is None:
    #         plt.savefig(path2fig)

    #     mean = np.mean(errors)
    #     std = np.std(errors)
    #     return (mean,std,np.percentile(errors,0),np.percentile(errors,100))

    def score(self,model, treshold,y_true):
        tp = [0,0]
        fp = [0,0]
        fn = [0,0]
        anomally_x,anomally_y = self.anomally_data
        if self.errors is None:
            
            try:           
                predicted_data = model.predict(anomally_x,verbose=0)
            except Exception as inst:
                            print(type(inst))    # the exception instance
                            print(inst.args)     # arguments stored in .args
                            print(inst) 

            predicted_data = predicted_data.reshape((-1,anomally_y.shape[-1]))
            all_errors = (anomally_x.reshape((-1,anomally_y.shape[-1])) - predicted_data)**2
            self.errors = tf.reduce_sum(all_errors,1)
            if not self.preproces_error is None:
                self.errors = self.preproces_error(self.errors)


        dic = {"A":True,"B":True,"C":True,'':True,'B1':True,'B2':True,'B3':True,'O':True,None:False,0:False,np.nan:False}
        y_true.fillna(0)        
        if self.y_visible is None:                   
            y_visible = []
            # try:
                
            for i,true in enumerate(y_true["Type"]):
                    # try:
                if y_true['Visible'][i] is True:
                        y_visible.append(dic[true])
                else:
                    y_visible.append(0)
            #         except Exception as inst:
            #             y_visible.append(0)
            #             print(inst)
            # except Exception as inst:
            #     print(inst)
            self.y_visible = y_visible
        if self.y_all is None:
            y_all = [dic[true] for i,true in enumerate(y_true["Type"]) ]
            self.y_all = y_all
        y_pred = self.errors>treshold
        # f14all,prec4all,rec4all = self.p_r_f1(self.y_all,y_pred)
        f14visible,prec4cisible,rec4visible = self.p_r_f1(self.y_visible,y_pred)
        return 0,0,0,f14visible,prec4cisible,rec4visible

    def visualisation(self,y_true,y_pred,treshold):
        if self.errors is None:
            errors = tf.reduce_sum((y_true-y_pred)**2,1)
        else:
            errors = self.errors
        anomalies = np.array([[i,err] for i,err in enumerate(errors) if err>treshold])
        normals = np.array([[i,err] for i,err in enumerate(errors) if err<treshold])
        anomalies = np.transpose(anomalies)
        normals = np.transpose(normals)
        plt.scatter(anomalies[0],anomalies[1],color = 'r')
        plt.scatter(normals[0],normals[1],color = 'g')
        plt.scatter(range(len(self.y_all)), self.y_all)
        plt.show()

    def visualise(self,model,treshold):
        anomally_x,anomally_y = self.anomally_data
        y_true = anomally_x.reshape((-1,anomally_x.shape[-1]))
        y_pred = predicted_data = model.predict(anomally_x,verbose=0).reshape((-1,anomally_y.shape[-1]))
        self.visualisation(y_true,y_pred,treshold)