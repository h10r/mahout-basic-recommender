package fi.aalto.recommenderSystems.recommender;

import java.io.File;
import java.io.IOException;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.common.RandomUtils;

public class UserBasedRecommenderWithEvaluation {

	/**
	 * Collaborative Filtering based on items using Mahout
	 * @param args
	 * @throws IOException 
	 */
	
	public static void main(String[] args) throws IOException {
		RandomUtils.useTestSeed();
		
		System.out.println( "UserBasedRecommenderWithEvaluation" );
		
		try {
			DataModel model = new FileDataModel( new File( "data/train.csv" ) );
						
			RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
			
			RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
				@Override
				public Recommender buildRecommender(DataModel model) throws TasteException {
					UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
										
					UserNeighborhood neighborhood = new NearestNUserNeighborhood(100, similarity, model); 
					return new GenericUserBasedRecommender(model, neighborhood, similarity);
				} 
			};
			
			double score = evaluator.evaluate(recommenderBuilder, null, model, 0.7, 1.0);
			System.out.println("SCORE with trainingPercentage 0.7 and evaluationPercentage 1.0: " + score);			
			
		} catch (IOException e) {
			System.out.println("Error loading file");
			e.printStackTrace();
		} catch ( TasteException e ) {
			System.out.println("Error Mahout");
			e.printStackTrace();
		}
		
	}

}
