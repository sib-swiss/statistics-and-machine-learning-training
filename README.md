webpage with course description : https://www.sib.swiss/training/course/2020-12-statsML



Tentative program :

	* python warm-up : read data , summarize it briefly, plot it, and perform simple stat tests on it
		-> use the dataset we used in the previous course / course exam

		course + practical


	* we want to focus on statistical learning, loosely defined as 
		"Statistical learning theory deals with the problem of finding a predictive function based on data." (wikipedia)
		and even more loosely defined as an articulation between stats and ML. 
		I would say we want to focus on interpretable models (ie. not black box "this classifies but we have no idea why or based on what features")


	* metrics and summary statistics of classification problems <- we can introduce this when we introduce classification .



stats :
		* linear :
			* OLS 

				key concepts :
					* tests / models hypothesis
					* parameter estimation
					* model comparison
					* overfitting ?
					-> I would say most of these can be introduced with a bit more details on the OLS.
						Then we can extend the approach 
					* model choice : contrast ML and stat approach

				datasets : 
					1 with high dimensionality to show (mtcars ? something biological ? )
					+ another one, maybe simpler for exercices 

			( * mixed linear models ) + difference entre time series and "classical" data

			* logistic regression ( + generalized linear models )

				key concepts :
					* link OLS to GLM 
					* model comparison : logit or probit ?



					-> use logistic regression to move to classification : can we say we "switch target" by putting a higher focus on prediction tasks
						* classification pipeline
						* classification evaluation / model choice -> compare with previous approach

				datasets:
					* different ecotoxicology datasets that show different cases (continuous response, multiple covariates, binary response) https://journals.plos.org/plosone/article/file?type=supplementary&id=info:doi/10.1371/journal.pone.0146021.s001#cite.inderjit%26streibig%26olofsdotter%3A2002
					* heart disease data : https://www.kaggle.com/ronitf/heart-disease-uci (there are a couple operations to perform to make this one practical)

	Then, we switch to an ML approach of classification -> at the end we come back to regression, but with an ML approach.


		* arbre decisionnel : random forest (boosted gradient -> to mention , intended for regression )
			* maybe an interesting point is to show that we now lack an explicit linear model, but the classification pipeline and tool still allow us to evaluate our tool and derive meaning from it (feature selection)

			dataset : TBD



