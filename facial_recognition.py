from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
from skimage.transform import rescale, resize
from numpy.linalg import matrix_rank
from skimage import color
from skimage import io

 
dataset_path = 'Dataset/'
dataset_dir  = os.listdir(dataset_path)
#No scaling
scaling_fact =1
width  = 195
height = 231
dim=width*height 
#scaling_fact=0.125
# width  = 49
# height = 58
# dim=58*49
# scaling_fact=0.5
# width  = 98
# height = 116
# dim=width*height 

##########    
# Training 
#########
   
print('-----Training Images-------')
train_image_names = ['subject01.normal.jpg', 'subject02.normal.jpg', 'subject03.normal.jpg', 'subject07.normal.jpg', 'subject10.normal.jpg', 'subject11.normal.jpg', 'subject14.normal.jpg', 'subject15.normal.jpg']
training_tensor   = np.ndarray(shape=(len(train_image_names), height*width), dtype=np.float64)
for i in range(len(train_image_names)):
    img_orig = plt.imread(dataset_path + train_image_names[i])
    #rescale images by a scaling factor 1/4 to get smaller images to process wih.
    img=rescale(img_orig, scaling_fact)
    training_tensor[i,0:dim] = np.array(img, dtype='float64').flatten()
    plt.subplot(2,4,1+i)
    plt.imshow(img, cmap='gray')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.title('Resized training images')
plt.show()


##########    
# Test
#########    
print('-----Test Images-------')
test_image_names = dataset_dir
#[i for i in dataset_dir if i not in train_image_names]
testing_tensor   = np.ndarray(shape=(len(test_image_names), height*width), dtype=np.float64)
for i in range(len(test_image_names)):
    img_orig = imread(dataset_path + test_image_names[i])
    #rescale images by a scaling factor 1/4 to get smaller images to process wih.
    img=rescale(img_orig, scaling_fact)
    testing_tensor[i,:] = np.array(img, dtype='float64').flatten()
    plt.subplot(3,6,1+i)
    plt.title(test_image_names[i].split('.')[0][-2:]+test_image_names[i].split('.')[1])
    plt.imshow(img, cmap='gray')
    plt.subplots_adjust(right=1.2, top=1.2)
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()
  
   
###########   
# Mean Face 
###########

print('-----Mean Face-------')
#mean_face = np.zeros((1,dim))
#average over the persons (rows) of all the variables (columns)
#mean_face=np.mean(training_tensor,dtype=np.float64,axis=0)
print('Mean over the rows')
mean_face = np.zeros((1,height*width))

for i in training_tensor:
    mean_face = np.add(mean_face,i)
    
mean_face = np.divide(mean_face,float(len(train_image_names))).flatten() 

plt.imshow(mean_face.reshape(height, width), cmap='gray')
plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.title('Mean face')
plt.show()
 
###########   
# Zero mean faces
###########

print('-----Zero mean training faces -------')
normalised_training_tensor = np.ndarray(shape=(len(train_image_names), height*width))

for i in range(len(train_image_names)):
    normalised_training_tensor[i] = np.subtract(training_tensor[i],mean_face)
   
### Display zero mean faces

for i in range(len(train_image_names)):
    img = normalised_training_tensor[i].reshape(height,width)
    plt.subplot(2,4,1+i)
    plt.imshow(img, cmap='gray')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.title('zero mean training faces')
plt.show()
 
########### 
## Covariance matrix
########### 

print('-----Covariance matrix------')
print('Warning! exploit the linear algebra tip')
print('Covariance matrix along ROWS ')
#Rigorously, cov matrix over the columns corresponds to rowvar= False
#with the linear algebra tip, it is enough to compute the matrix over ROWS
 
cov_matrix = np.cov(normalised_training_tensor, rowvar= True)

#Check the rank of the matrix is 7 as the sum of the variables is zero 
print(matrix_rank(cov_matrix))
#print(cov_matrix.shape)

cov_matrix = np.divide(cov_matrix,len(train_image_names))
print('Covariance matrix of X: %s' %cov_matrix)

########### 
## Eigenvalues and eigenvectors
########### 

eigenvalues_orig, eigenvectors, = np.linalg.eig(cov_matrix)
eigenvalues=eigenvalues_orig.real
print('Eigenvectors of Cov(X): %s' %eigenvectors)
print('Eigenvalues of Cov(X): %s' %eigenvalues)
 
eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]
# Sort the eigen pairs in descending order
eig_pairs.sort(reverse=True)
eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenvalues))]
eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]

########### 
## Scree-graph
########### 
plt.title('----Scree-graph')  
plt.xlabel('----Principal Components')
plt.ylabel('----Eigenvalue')
num_comp = range(1,len(eigenvalues)+1)
plt.plot(num_comp, eigvalues_sort )
plt.show()
########### 
## Find cumulative variance of each principle component
########### 
  
  
var_comp_sum = np.cumsum(eigvalues_sort)/sum(eigvalues_sort)
# Show cumulative proportion of variance with respect to components
print('----Cumulative proportion of variance explained vector: %s' %var_comp_sum)
# x-axis for number of principal components kept
num_comp = range(1,len(eigvalues_sort)+1)
plt.title('----Cum. Prop. Variance Explain and Components Kept')
plt.xlabel('----Principal Components')
plt.ylabel('Cum. Prop. Variance Explained')
plt.scatter(num_comp, var_comp_sum)
plt.show()
   
   
#######################################   
## Choose the necessary no. of principle components
#####################################

dim_reduced=5
reduced_data = np.array(eigvectors_sort[:dim_reduced]).transpose()


proj_data = np.dot(training_tensor.transpose(),reduced_data)
proj_data = proj_data.transpose()
print(proj_data.shape)

####################################### 
## Plot eigen faces
#######################################

print('-----plot eigen faces-----')
for i in range(proj_data.shape[0]):
    img = proj_data[i].reshape(height,width)
    plt.subplot(2,4,1+i)
    plt.imshow(img, cmap='jet')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.title('Eigen faces')
plt.show()

#######################################   
## Finding weights for each training image
#######################################  
    
w = np.array([np.dot(proj_data,i) for i in normalised_training_tensor])
print('----Weights for each training image')
w
      
#######################################   
## Recognizing an unknown face
#######################################     

print('----Read and display an unknown face')
unknown_face        = plt.imread('Dataset/subject12.normal.jpg')
unknown_face_vector = np.array(unknown_face, dtype='float64').flatten()
plt.imshow(unknown_face, cmap='gray')
plt.title('Unknown face')
plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()
  
#######################################    
## Normalise unknown face"
#######################################   

print('----Normalise unknown face')
normalised_uface_vector = np.subtract(unknown_face_vector,mean_face)
plt.imshow(normalised_uface_vector.reshape(height, width), cmap='gray')
plt.title('Normalised unknown face')
plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()
 
 
#######################################    
## Weights of unknown face
#######################################  

print('----Project unkown face on principal components')
w_unknown = np.dot(proj_data, unknown_face_vector)
print('Weights of unknown face')
w_unknown
  
#######################################    
## Find the best matching
#######################################  

print('----Find the best matching  min|W - W_{unknown}|----') 
diff  = w - w_unknown
norms = np.linalg.norm(diff, axis=1)
min_value_norms = min(norms)

min_index_norms =np.argmin(norms, axis=0)
print('Best match unknown with training subject %s' %min_index_norms)

   
#########################################    
## Recognition of  all test images"
 #########################################     
  
   
count        = 0
num_images   = 0
correct_pred = 0
def recogniser(img, train_image_names,proj_data,w):
    global count,highest_min,num_images,correct_pred
    unknown_face        = plt.imread('Dataset/'+img)
    num_images          += 1
    unknown_face_vector = np.array(unknown_face, dtype='float64').flatten()
    normalised_uface_vector = np.subtract(unknown_face_vector,mean_face)

    plt.subplot(9,4,1+count)
    plt.imshow(unknown_face, cmap='gray')
    plt.title('Input:'+'.'.join(img.split('.')[:2]))
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
    count+=1
    
    w_unknown = np.dot(proj_data, normalised_uface_vector)
    diff  = w - w_unknown
    norms = np.linalg.norm(diff, axis=1)
    index = np.argmin(norms)

    t1 = 100111536
    #t1 = 200535910.268 
    # working with 6 faces
    #t0 = 86528212
    t0 = 88831687
    #t0 = 143559033 
    # working with 6 faces
    
    if norms[index] < t1:
        plt.subplot(9,4,1+count)
        if norms[index] < t0: # It's a face
            if img.split('.')[0] == train_image_names[index].split('.')[0]:
                plt.title('Matched:'+'.'.join(train_image_names[index].split('.')[:2]), color='g')
                plt.imshow(imread('Dataset/'+train_image_names[index]), cmap='gray')
                
                correct_pred += 1
            else:
                plt.title('Matched:'+'.'.join(train_image_names[index].split('.')[:2]), color='r')
                plt.imshow(imread('Dataset/'+train_image_names[index]), cmap='gray')
        else:
            if img.split('.')[0] not in [i.split('.')[0] for i in train_image_names] and img.split('.')[0] != 'apple':
                plt.title('Unknown face!', color='g')
                correct_pred += 1
            else:
                plt.title('Unknown face!', color='r')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
        plt.subplots_adjust(right=1.2, top=2.5)
    else:     
        plt.subplot(9,4,1+count)
        if len(img.split('.')) == 3:
            plt.title('Not a face!', color='r')
        else:
            plt.title('Not a face!', color='g')
            correct_pred += 1
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
    count+=1
    
fig = plt.figure(figsize=(15, 15))
for i in range(len(test_image_names)):
    recogniser(test_image_names[i], train_image_names,proj_data,w)

plt.show()
print('Correct predictions: {}/{} = {}%'.format(correct_pred, num_images, correct_pred/num_images*100.00))


