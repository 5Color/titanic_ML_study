# titanic_ML_study

타이타닉 데이터를 이용한 생존자 분류예측 모델을 만들어보았다.

# 사용 데이터:

우선 사용된 데이터는 titanic.csv으로 891 rows × 12 columns이다.

# 전처리
우선 첫번째로 numpy, pandas, LabelEncoder, train_test_split과 같은 라이브러리들을 import해준다.
그 다음으로 학습에 사용할 데이터 컬럼으로 ['Sex','Age','SibSp','Parch','Survived','Embarked']을 사용하기 위해 .dropna()를 진행하여 NaN값을 없애주고 df변수에 담는다.
x = df[['Sex','Age','SibSp','Parch','Embarked']]이고
y = df['Survived']으로 Survived가 종속변수로 이진분류가 된다

Sex와 Embarked컬럼은 숫자형 데이터가 아니니 전처리를 해줘야하는데 이때, LabelEncoder을 사용하여 원핫인코딩 해주고, 원핫인코딩된 데이터를 기존 데이터변수에 덮어넣어버린다

이제 train_test_split을 통해 데이터셋을 분리하고 train_input.shape을 찍어 확인해보았더니 (569, 5)라는 결과값이 나왔다.
여기서 알 수 있는 사실은 input데이터은 2차원이고, 5개의 feature을 가지고 있다는것이다. (입력층)

# 모델학습 및 결과:

오늘 학습에 이용할 모델들은 scikit-learn의 [ KNeighborsClassifier, DecisionTreeClassifier, LogisticRegression, tensorflow의 인공신경망, boosting계열의 앙상블기법 모델 lightgbm의 LGBMClassifier]이다

첫번째 학습할 모델은 KNeighborsClassifier()이다. 매개변수로 (n_neighbors=3)로 설정하고 train_input과 train_target을 fitting해준다.
<결과값>
ㄴ Train Score: 0.8506151142355008
ㄴ Test Score: 0.6293706293706294
너무 과적합 되어있는것을 알 수 있다

두번째 학습할 모델은 DecisionTreeClassifier()이다. 매개변수 설정값은 비워둔다. 
<결과값>
ㄴ Train Score: 0.9314586994727593
ㄴ Test Score: 0.7132867132867133
이녀석도 너무 과적합 되어있다


세번째 학습할 모델은 LogisticRegression()이다. 매개변수 설정값은 비워둔다.
<결과값>
ㄴ Train Score: 0.7926186291739895
ㄴ Test Score: 0.7482517482517482


네번째는 인공신경망을 활용한 학습이다. keras를 쓸것이고, input_shape는 5,로 설정, 은닉층을 2층, 활성화함수 'relu'로 설정, 마지막 출력층으로 sigmoid함수를 쓴다. ( sigmoid <=이진출력임 ) <-- 신경망 구축
옵티마이저는 Nadam, 손실함수는 'binary_crossentropy'로 설정한다 <-- 모델설정
epochs은 1000으로 잡고 학습을 시켜준다.

<결과값>
학습할때마다 다른지만 내가 시각화를 해본 결과 대부분 잘 나온다.
model.evaluate(test_input, test_target) : accuracy: 0.8252 - loss: 0.4768 이다
2번째로 사용한 모델중 성능이 잘 나왔다


마지막으로 lightGBM의  LGBMClassifier()모델을 이용해 학습해준다.
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score, roc_auc_score
불러 예측해준다. 결과값을 보았을때 이 모델이 가장 성능이 우수한것으로 추정된다.
<결과값>
ㄴ recall_score :  0.6935483870967742
ㄴ roc_auc_score :  0.7541816009557947
ㄴ precision_score :  0.7413793103448276
ㄴ accuracy_score :  0.7622377622377622
ㄴ f1_score :  0.7166666666666667


