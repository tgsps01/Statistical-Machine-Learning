import numpy as np
import scipy.io

# Load the .mat files
train_0_data = scipy.io.loadmat('./Desktop/SML/train_0_img.mat')['target_img']
train_1_data = scipy.io.loadmat('./Desktop/SML/train_1_img.mat')['target_img']
test_0_data = scipy.io.loadmat('./Desktop/SML/test_0_img.mat')['target_img']
test_1_data = scipy.io.loadmat('./Desktop/SML/test_1_img.mat')['target_img']

train_0_data = np.transpose(train_0_data, (2, 0, 1))/255.0
train_1_data = np.transpose(train_1_data, (2, 0, 1))/255.0

test_0_data = np.transpose(test_0_data, (2, 0, 1))/255.0
test_1_data = np.transpose(test_1_data, (2, 0, 1))/255.0

train_0_data.shape

train_1_data.shape

def extract_features(data):
    feature_1 = []
    feature_2 = []   
    for img in data:
        avg_pixel_value = np.mean(img)
        std_pixel_value = np.std(img)
        feature_1.append(avg_pixel_value)
        feature_2.append(std_pixel_value)
    mean_feature1 = np.mean(feature_1)
    mean_feature2 = np.mean(feature_2)
    std_feature1 = np.std(feature_1)
    std_feature2 = np.std(feature_2)
    
    return {'mean_feature1': mean_feature1,
        'mean_feature2': mean_feature2,
        'std_feature1': std_feature1,
        'std_feature2': std_feature2}

# Extract features for training data
train_0_features = extract_features(train_0_data)
train_1_features = extract_features(train_1_data)

mean_feature1_train0 = train_0_features['mean_feature1']
std_feature1_train0 = train_0_features['std_feature1']
mean_feature2_train0 = train_0_features['mean_feature2']
std_feature2_train0 = train_0_features['std_feature2']

mean_feature1_train1 = train_1_features['mean_feature1']
std_feature1_train1 = train_1_features['std_feature1']
mean_feature2_train1 = train_1_features['mean_feature2']
std_feature2_train1 = train_1_features['std_feature2']

def gaussian_pdf(m, std, f):
    # plug in the Gaussian PDF formula, given mean, variance, and a specific feature value x_f
    # applies to both feature1 and feature2
    result = 1/(np.sqrt(2*np.pi * std**2)) * np.exp((-0.5/(std**2)) * np.square(f-m))
    return result

total_0_sample = train_0_data.shape[0]
total_1_sample = train_1_data.shape[0]
p_of_0 = total_0_sample/(total_0_sample+total_1_sample)
p_of_1 = total_1_sample/(total_0_sample+total_1_sample)

correct0 = 0
correct1 = 0
for x in test_0_data:
    x_f1 = np.mean(x)
    x_f2 = np.std(x)
    # obtain prediction and see if it predicts 0
    predict0 = gaussian_pdf(mean_feature1_train0, std_feature1_train0, x_f1) * gaussian_pdf(mean_feature2_train0, std_feature2_train0, x_f2) * p_of_0
    predict1 = gaussian_pdf(mean_feature1_train1, std_feature1_train1, x_f1) * gaussian_pdf(mean_feature2_train1, std_feature2_train1, x_f2) * p_of_1
    if (predict0 > predict1):
        correct0 += 1
acc0 = correct0 / len(test_0_data)
for x in test_1_data:
    x_f1 = np.mean(x)
    x_f2 = np.std(x)
    # obtain prediction and see if it predicts 1
    predict0 = gaussian_pdf(mean_feature1_train0, std_feature1_train0, x_f1) * gaussian_pdf(mean_feature2_train0, std_feature2_train0, x_f2) * p_of_0
    predict1 = gaussian_pdf(mean_feature1_train1, std_feature1_train1, x_f1) * gaussian_pdf(mean_feature2_train1, std_feature2_train1, x_f2) * p_of_1
    if (predict0 < predict1):
        correct1 += 1
acc1 = correct1 / len(test_1_data)
print([acc0, acc1])