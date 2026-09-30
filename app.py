from __future__ import annotations
import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import plotly.graph_objects as go

TITLE='Hybrid Quantum-Classical Machine Learning Platform'
st.set_page_config(page_title=TITLE,page_icon='⚛️',layout='wide')
st.markdown('''<style>.stApp{background:#0B1220}.block-container{max-width:1400px}.card{padding:16px;border:1px solid #2B3C52;border-radius:12px;background:#121D2E}</style>''',unsafe_allow_html=True)

@st.cache_data
def sample_data():
    d=load_breast_cancer(as_frame=True)
    X=d.data.copy(); y=pd.Series(d.target_names[d.target],name='Diagnosis')
    return X,y

X,y=sample_data()

st.sidebar.title('Research workspace')
st.sidebar.caption('Biomedical AI · Hybrid quantum/classical experiments')
page=st.sidebar.radio('Workflow',['Dashboard','Dataset','Preprocessing','Feature Engineering','Classical ML','Quantum ML','Explainability','Benchmark','Results'])
st.sidebar.info('Research/education only. Predictions are not medical diagnoses.')

positive_options=sorted(y.unique().tolist())
pos_default='malignant' if 'malignant' in positive_options else positive_options[-1]
positive=st.sidebar.selectbox('Positive class',positive_options,index=positive_options.index(pos_default))
test_size=st.sidebar.slider('Held-out test fraction',0.15,0.35,0.30,0.05)
k=st.sidebar.slider('Features retained',1,X.shape[1],10)
ncomp=st.sidebar.slider('PCA components / qubits',1,min(4,k),min(4,k))
steps=st.sidebar.slider('VQC optimizer steps',2,20,8)

@st.cache_data
def run_models(test_size,k,ncomp,positive,steps):
    yb=np.array([1 if v==positive else 0 for v in y])
    idx=np.arange(len(yb)); tr,te=train_test_split(idx,test_size=test_size,random_state=42,stratify=yb)
    imp=SimpleImputer(strategy='median'); scaler=StandardScaler()
    Xt=scaler.fit_transform(imp.fit_transform(X.iloc[tr])); Xv=scaler.transform(imp.transform(X.iloc[te]))
    selector=SelectKBest(f_classif,k=k); Xts=selector.fit_transform(Xt,yb[tr]); Xvs=selector.transform(Xv)
    selected=X.columns[selector.get_support()].tolist()
    pca=PCA(n_components=ncomp,random_state=42); Pt=pca.fit_transform(Xts); Pv=pca.transform(Xvs)
    svm=SVC(C=1.0,kernel='rbf',probability=True,random_state=42)
    t=time.perf_counter(); svm.fit(Pt,yb[tr]); train_time=time.perf_counter()-t
    t=time.perf_counter(); sp=svm.predict(Pv); infer=time.perf_counter()-t
    ss=svm.predict_proba(Pv)[:,1]
    def metrics(pred,score,train_t=None,infer_t=None):
        return {'Accuracy':accuracy_score(yb[te],pred),'Precision':precision_score(yb[te],pred,zero_division=0),'Recall / Sensitivity':recall_score(yb[te],pred,zero_division=0),'Specificity':specificity(yb[te],pred),'F1 Score':f1_score(yb[te],pred,zero_division=0),'ROC-AUC':roc_auc_score(yb[te],score),'Training Time (s)':train_t,'Inference Time (ms/sample)':infer_t*1000/max(1,len(te))}
    svm_metrics=metrics(sp,ss,train_time,infer)
    quantum=None; qmetrics=None; qerror=None
    try:
        import pennylane as qml
        dev=qml.device('default.qubit',wires=ncomp)
        def circuit(weights,angles):
            qml.AngleEmbedding(angles,wires=range(ncomp),rotation='Y')
            qml.BasicEntanglerLayers(weights,wires=range(ncomp))
            return qml.expval(qml.PauliZ(0))
        qnode=qml.QNode(circuit,dev)
        rng=np.random.default_rng(42); weights=rng.normal(0,0.1,(1,ncomp)); opt=qml.AdamOptimizer(stepsize=0.08)
        def loss(w):
            vals=np.array([qnode(w,a) for a in np.tanh(Pt)*np.pi])
            return np.mean((vals-(2*yb[tr]-1))**2)
        t=time.perf_counter()
        for _ in range(steps): weights=opt.step(loss,weights)
        qtrain=time.perf_counter()-t
        t=time.perf_counter(); z=np.array([qnode(weights,a) for a in np.tanh(Pv)*np.pi]); qinfer=time.perf_counter()-t
        qp=(z<0).astype(int); qscore=-z
        qmetrics=metrics(qp,qscore,qtrain,qinfer); quantum={'n_qubits':ncomp,'n_layers':1,'weights':weights,'expectations':lambda angles: np.array([qnode(weights,angles)])}
    except Exception as e: qerror=f'{type(e).__name__}: {e}'
    preds=pd.DataFrame({'Actual Class':np.array(y.iloc[te]),'SVM Model Prediction':np.array([positive if v else next(c for c in positive_options if c!=positive) for v in sp])})
    if quantum:
        preds['VQC Model Prediction']=[positive if v else next(c for c in positive_options if c!=positive) for v in qp]
    return {'X_train':Pt,'X_test':Pv,'y_train':yb[tr],'y_test':yb[te],'selected':selected,'selector':selector,'pca':pca,'svm':svm,'svm_metrics':svm_metrics,'qmetrics':qmetrics,'quantum':quantum,'quantum_error':qerror,'predictions':preds,'test_indices':te,'all_components':np.vstack([Pt,Pv]),'all_labels':np.r_[yb[tr],yb[te]],'positive':positive,'negative':next(c for c in positive_options if c!=positive)}

def specificity(actual,pred):
    cm=confusion_matrix(actual,pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel(); return tn/(tn+fp) if tn+fp else 0.0

with st.spinner('Preparing the experiment...'):
    result=run_models(test_size,k,ncomp,positive,steps)

metrics={'Classical SVM':result['svm_metrics']}
if result['qmetrics']: metrics['Quantum VQC']=result['qmetrics']

st.title(TITLE)
st.caption('Same cleaned data → feature selection → PCA → classical SVM and optional PennyLane VQC → held-out benchmark')

if page=='Dashboard':
    a,b,c,d=st.columns(4)
    a.metric('Dataset rows',len(y)); b.metric('Source features',X.shape[1]); c.metric('Selected features',len(result['selected'])); d.metric('Test samples',len(result['y_test']))
    st.subheader('Hybrid workflow')
    st.write('Biomedical dataset → preprocessing → SelectKBest → PCA → SVM + VQC → explainability → benchmark')
    st.dataframe(pd.DataFrame(metrics).T[['Accuracy','Precision','Recall / Sensitivity','F1 Score','ROC-AUC']],width='stretch')
elif page=='Dataset':
    st.subheader('Breast Cancer · Wisconsin Diagnostic')
    st.write('Bundled demonstration dataset loaded from scikit-learn. It contains 569 observations, 30 numeric predictors and a binary diagnosis target.')
    st.dataframe(pd.concat([X.head(10),y.head(10)],axis=1),width='stretch')
    st.bar_chart(y.value_counts())
elif page=='Preprocessing':
    st.subheader('Preprocessing')
    st.write('Missing numeric values are median-imputed using the training partition only, then standardized before feature selection and PCA.')
    st.json({'Train samples':len(result['y_train']),'Test samples':len(result['y_test']),'PCA components':ncomp,'Positive class':positive,'Negative class':result['negative']})
elif page=='Feature Engineering':
    st.subheader('Feature Engineering')
    scores=pd.Series(result['selector'].scores_,index=X.columns).sort_values(ascending=False)
    st.dataframe(scores.head(k).rename('ANOVA F score').to_frame(),width='stretch')
    st.write('Selected features:',', '.join(result['selected']))
elif page=='Classical ML':
    st.subheader('Classical ML · SVM')
    st.dataframe(pd.DataFrame(result['svm_metrics'],index=['Value']).T,width='stretch')
    cm=confusion_matrix(result['y_test'],result['svm'].predict(result['X_test']),labels=[0,1]); st.write('Confusion matrix'); st.dataframe(pd.DataFrame(cm,index=['Actual negative','Actual positive'],columns=['Pred negative','Pred positive']))
elif page=='Quantum ML':
    st.subheader('Quantum ML · PennyLane VQC')
    if result['quantum'] is None:
        st.warning('The classical path is available, but the quantum simulator could not be executed in this deployment.')
        st.code(result['quantum_error'] or 'No simulator error returned.')
    else:
        st.success(f"VQC executed with {result['quantum']['n_qubits']} qubits on PennyLane default.qubit.")
        st.dataframe(pd.DataFrame(result['qmetrics'],index=['Value']).T,width='stretch')
elif page=='Explainability':
    st.subheader('Explainability')
    scores=pd.Series(result['selector'].scores_,index=X.columns).sort_values(ascending=False).head(k)
    st.bar_chart(scores)
    st.caption('Feature scores are training-partition ANOVA F scores. They are statistical feature-selection values, not clinical importance claims.')
elif page=='Benchmark':
    st.subheader('Classical vs Quantum Benchmark')
    st.dataframe(pd.DataFrame(metrics).T,width='stretch')
    fig=go.Figure()
    for name,m in metrics.items(): fig.add_bar(name=name,x=['Accuracy','Precision','Recall','F1','ROC-AUC'],y=[m['Accuracy'],m['Precision'],m['Recall / Sensitivity'],m['F1 Score'],m['ROC-AUC']])
    fig.update_layout(template='plotly_dark',barmode='group',yaxis_range=[0,1]); st.plotly_chart(fig,width='stretch')
elif page=='Results':
    st.subheader('Held-out Results')
    st.dataframe(result['predictions'],width='stretch')
    st.caption('These are research-model predictions on a held-out demonstration dataset, not medical diagnoses.')
    if result['quantum_error']: st.info('Quantum path note: '+result['quantum_error'])

st.divider(); st.caption('Research/education only · No clinical diagnosis or treatment recommendation.')
