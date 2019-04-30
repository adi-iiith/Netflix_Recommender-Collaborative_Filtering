# Team Houdini (33)

## [Netflix_Recommender-Collaborative_Filtering -- Yehuda Koren](https://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf)


### To-Do List

1.  CF deals with building recommendation systems like movie recommender on netflix based on your reviews and choices.
2.  Implement baseline CF models mentioned in paper(SVD based).
3.  Improve them using technique specified in paper.

### Status : Project Implemented.

### Dataset

Use dataset as " ../input/u.data " [Ml-100k](https://grouplens.org/datasets/movielens/100k/)

#### Overview : 
			
			.. Save the Dataset in "../input" folder.
			.. (python v3.x) 
			.. To train and test Integrated Model :

				--Train model                  -- " python3 integrated_model.py " 
				--Test  model 				   -- " python3 integrated_train.py " 
				( OR )
				-- Train and Test (per epoch ) -- " python3 integrated_model_epoch_error.py"

			.. To train and test Neighborhood model :
				-- Train and Test (per epoch ) -- " python3 neighborhood.py"

			.. To train and test SVDpp model :
				-- Train and Test (per epoch ) -- "python3 svdpp.py"

			.. Predictions for Integrated model (Demo)
				-- test_beta.ipynp

			.. EDA for Dataset :
				-- EDA.ipynp

			.. Project Report 

			.. Project Presentation 