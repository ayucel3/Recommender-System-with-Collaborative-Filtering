#Aral Yucel Data Mining Project B00721425
'''
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import GridSearchCV
from surprise import Reader
import pandas as pd #https://pandas.pydata.org/docs/reference/index.html all the methods I used I searched from here
from surprise import NMF
from surprise import SVDpp

#pulling the data from text file and making it a dataframe
data = pd.read_csv ('train.txt', header = None, sep=' ')
data.columns = ['User_ID', 'Movie_ID', 'Ratings']
reader = Reader(rating_scale=(1,5))
dataset = Dataset.load_from_df(data[['User_ID', 'Movie_ID', 'Ratings']], reader)

#https://surprise.readthedocs.io/en/stable/matrix_factorization.html I found the documentation for parameters for above methods here.
#from https://surprise.readthedocs.io/en/stable/getting_started.html I found how to do GridSearchCV
#from the codes below I tested SVD,NMF and SVDpp functions to see which one is giving
#the best resut for rmse value you can command and un command parts to see
#I took this code to be able to find the best parameters to run by SVD
#To be able to get best rmse possible this will help us predict ratings more accurate
#I changed the parameters a lot this is not the only set up I tried.
#I run the test for atleast 5 times to be able to make sure I choose the right parameters
'''
'''
name = 'SVD: '
#n_epochs: The number of iteration of the SGD procedure.
#lr_all: The learning rate for all parameters.
#reg_all: The regularization term for all parameters.
param_grid_SVD = {'n_epochs': [20], 'lr_all': [0.005],
              'reg_all': [0.02,0.1,0.05,0.2]}
#the code below splits the data to 5 randomly equal datasets and uses 1 of them as testset
#and others as training set to test and see which n_epochs,lr_all,reg_all values are the best
#choice for low rmse.
gs = GridSearchCV(SVD, param_grid_SVD, measures=['rmse'], cv=5)
'''

'''
name = 'NMF: '
param_grid_NMF = {'n_epochs': [20]}
gs = GridSearchCV(NMF, param_grid, measures=['rmse'], cv=5)
'''

'''
name = 'SVDpp: '
#n_epochs: The number of iteration of the SGD procedure.
#lr_all: The learning rate for all parameters.
#reg_all: The regularization term for all parameters.
param_grid_SVDpp = {'n_epochs': [20], 'lr_all': [0.005,0.007],
              'reg_all': [0.1,0.05]}
gs = GridSearchCV(SVD, param_grid_SVDpp, measures=['rmse'], cv=5)
'''
'''
gs.fit(dataset)

# best RMSE score
print(name + str(gs.best_score['rmse']))

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])
'''

'''
#I wanted to play with the test and train size between svdpp and svd to find which one is givin better rmse
import pandas as pd
from surprise import SVD
from surprise import SVDpp
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import Reader
from surprise import Dataset

data = pd.read_csv ('train.txt', header = None, sep=' ')
data.columns = ['User_ID', 'Movie_ID', 'Ratings']
reader = Reader(rating_scale=(1,5))
dataset = Dataset.load_from_df(data[['User_ID', 'Movie_ID', 'Ratings']], reader)
trainset, testset = train_test_split(dataset, test_size = .25) #https://surprise.readthedocs.io/en/stable/getting_started.html here is the doc I found for this part I figured out Train-test split method section

model = SVDpp(n_epochs= 20, lr_all= 0.007, reg_all= 0.1)

model.fit(trainset)

model_SVD = SVD(n_epochs = 20, lr_all = 0.005, reg_all = 0.1)
model_SVD.fit(trainset)

predictions = model.test(testset)
print('SVDpp')
accuracy.rmse(predictions)

predictions = model_SVD.test(testset)
print('SVD')
accuracy.rmse(predictions)
'''

#I found all the functions that I used from surprise library is here => https://surprise.readthedocs.io/en/v1.0.1/index.html
from surprise import SVDpp #After doing lots of testing on SVD, SVDpp and NMF
#I conclude that SVDpp is giving the best RMSE value with test and train splits
#Thats why I wanted to choose SVDpp for this project to predict values.
#you can check SVDpp algorithm from https://surprise.readthedocs.io/en/stable/matrix_factorization.html
from surprise import Dataset #to create dataset
import pandas as pd #pandas for dataframe #https://pandas.pydata.org/docs/reference/index.html all the methods I used I searched from here
from surprise import accuracy # to calculate rmse value
from surprise import Reader #Reader object for Surprise to be able to parse the file or the dataframe.

data = pd.read_csv ('train.txt', header = None, sep=' ')#reading train.txt file and converting it to dataframe
data.columns = ['User_ID', 'Movie_ID', 'Ratings'] #sicne the train.txt does not have column names we put the column names to dataframe
reader = Reader(rating_scale=(1,5))# A reader is still needed but only the rating_scale param is requiered.
dataset = Dataset.load_from_df(data[['User_ID', 'Movie_ID', 'Ratings']], reader)#Creating dataset https://surprise.readthedocs.io/en/stable/dataset.html

#I made this matrix to make a condition in output phase.If the user rated the movie we dont want to make prediction again we only need to pull the
#rating that user did for that spesific movie thats why this matrix helped me pull data I need from train.txt
user_item_matrix = data.pivot_table(index= 'User_ID', columns= 'Movie_ID', values = "Ratings")

print('Data Frame Created')


#finding the biggest user_ID to be able to reach the last index of user_id list
user_id_list = data.User_ID.unique() 
user_id_list.sort()

#finding the bigget movie_id to be able to reach the last index of movie_id
movie_id_list = data.Movie_ID.unique()
movie_id_list.sort()

model = SVDpp(n_epochs = 20, lr_all = 0.007, reg_all = 0.1)#selecting model with spesificed parameters

print('Fitting the model to dataset')
model.fit(dataset.build_full_trainset())#fitting dataset to model

print('Creating the submit_sample.txt')
file = open("submit_sample.txt","w")#creating submit_sample.txt as output
for user_id in range(user_id_list[-1]):#user_id_list[-1] is equal to 943 and user id loops from 0 to 942 in this case
    for movie_id in range(1,movie_id_list[-1]+1):#for loop goes from (1 to 1682) in this case 
        #I did try Except here cause in user_item_matrix there are some missing columns which will create error when we write
        #(user_item_matrix.iloc[user_id][movie_id]) since there is no data about the missing columns
        #Since my model can predict everycolumn eventhough it is missing I wrote except: part to just take out the predictions of
        #those missing columns to output file. This way I did not need to create the actual columns for the missing data.
        #Instead I output the prediction from model.predict to output file.
        try:
            #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html for iloc[][] usage.
            if pd.isnull(user_item_matrix.iloc[user_id][movie_id]):#user_item_matrix.iloc[0][1]  means first users rating for 1 movie
                
                #model.predict documentation https://surprise.readthedocs.io/en/stable/getting_started.html
                #predict(1,1) means the prediction that user 1 made for movie 1 thats why I wrote user_id+1
                pred = model.predict(user_id+1, movie_id).est #doing prediction with SVDpp model for each user to each movie one by one

                if type(pred) == int:#if the prediction is int. It gives error I put the condition to handle the error
                    rating = pred
                else:
                    rating = pred.round()
                rating = int(rating)#to be able to put integers instead of floats
                
            else:#if the user already gave rating to movie we just take that instead of predicting the rating.
                rating = int(user_item_matrix.iloc[user_id][movie_id])
      
        except:# if the movie does not exists in dataframe the codes gives error so instead I handled the error by making the prediction and insert it to rating value for that spesific user to non existed movie. 
            pred = model.predict(user_id+1, movie_id).est #doing prediction with SVDpp model for each user to each movie one by one
            if type(pred) == int:#if the prediction is int. It gives error I put the condition to handle the error
                rating = pred
            else:
                rating = pred.round()
            rating = int(rating)#to be able to put integers instead of floats
            
        #Writing all the predictions to submit_sample.txt file line by line.
        file.write(str(user_id+1) + ' ' + str(movie_id) + ' ' + str(rating) + "\n")  
file.close()
print('File created you can see it from the folder!!')






