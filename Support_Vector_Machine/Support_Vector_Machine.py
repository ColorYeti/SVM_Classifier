import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm
style.use("ggplot")

class SVM:
    def __init__(self):
        self.colors={1:'r',-1:'b'}
        self.fig=plt.figure()
        self.ax=self.fig.add_subplot(1,1,1)

    def fit(self,data):
        self.data=data
        dict={}
        all_data=[]
        b_range_multiple=2
        b_multiple=5
        transforms=[[1,1],
                    [-1,1],
                    [-1,-1],
                    [1,-1]]
        

        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value=max(all_data)
        self.min_feature_value=min(all_data)
        all_data=None
        step_sizes=[self.max_feature_value*0.1,self.max_feature_value*0.01,self.max_feature_value*0.001]
        latest_optimum=self.max_feature_value*10

        for step in step_sizes:
            w=np.array([latest_optimum,latest_optimum])
            optimized=False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),self.max_feature_value*b_range_multiple,step*b_multiple):
                    for transformations in transforms:
                        w_t=w*transformations
                        found_options=True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b)>=1:
                                    found_options=False
                        if found_options:
                            dict[np.linalg.norm(w_t)]=[w_t,b]


                if w[0]<0:
                    optimized=True
                else:
                    w=w-step

            norms=sorted([n for n in dict])

            choice = dict[norms[0]]
            self.w=choice[0]
            self.b=choice[1]
            latest_optimum=choice[0][1]+step*2
        for i in self.data:
            for xi in self.data[i]:
                yi=i
                print(xi,':',yi*(np.dot(self.w,xi)+self.b))


    def projection(self,x):
        pass

    def prediction(self,features):
        classification=np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification!=0:
            self.ax.scatter(features[0],features[1],200,'*',self.colors[classification])
        return classification

    def visualise(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data[i]]for i in data]
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v)/w[1]

        datarange=self.min_feature_value*0.9,self.max_feature_value*1.1
        hyp_x_max=datarange[1]
        hyp_x_min=datarange[0]
        psv1=hyperplane(hyp_x_max,self.w,self.b,1)
        psv2=hyperplane(hyp_x_min,self.w,self.b,1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2],'k')
        nsv1=hyperplane(hyp_x_max,self.w,self.b,-1)
        nsv2=hyperplane(hyp_x_min,self.w,self.b,-1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2],'k')
        sv1=hyperplane(hyp_x_max,self.w,self.b,0)
        sv2=hyperplane(hyp_x_min,self.w,self.b,0)
        self.ax.plot([hyp_x_min,hyp_x_max],[sv1,sv2],'k')

        plt.show()

            

data={-1:np.array([[1,7],
                  [2,8],
                  [3,8],]),
      1:np.array([[5,1],
                 [6,-1],
                 [7,3], ])}

svm = SVM()
svm.fit(data)
svm.visualise()