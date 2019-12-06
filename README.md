# Twitter_2
Use Embeddings and and LSTM to create a metric on twitter accounts

Much of the code for setting up the LSTM was orginally copied from https://github.com/WillKoehrsen/recurrent-neural-networks/tree/master/notebooks

The idea in this project is to use two embedding layers, one for the words, and then another for the accounts.  The idea is that by telling the LSTM what account had created the tweet, the LSTM would find this useful in predicting the next word.  Of course, we don't give an account, instead we give a vector, and hope that the training will bring accounts that have similar language structure closer together.  So this *should* work better than Bag of Word methods. 

I ran this for a little while and the results seem promising - you can see some lookalike accounts found in the "SomeResults.txt"


If I get a chance I will organize the cleaning and prepping of the training file. My first time through the method was roundabout and the training file was far too large to upload.    

I'm hoping to run some experiments where I have the same users, split into different users separated by weeks, and test if the vector embeddings agree with each other.   Beyond that I would like to try to do this in a way that is independent of word bags.  I'm also hoping to run these experiments several times and see if the end result depends on the random initialization or not.  
