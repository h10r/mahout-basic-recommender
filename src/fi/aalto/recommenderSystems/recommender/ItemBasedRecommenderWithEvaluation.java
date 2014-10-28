package fi.aalto.recommenderSystems.recommender;

import java.io.File;
import java.io.IOException;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.common.RandomUtils;

public class ItemBasedRecommenderWithEvaluation {

	/**
	 * Collaborative Filtering based on items using Mahout
	 * @param args
	 * @throws IOException 
	 */
	
	public static void main(String[] args) throws IOException {
		RandomUtils.useTestSeed();
		
		System.out.println( "ItemBasedRecommenderWithEvaluation" );
		
		try {
			DataModel model = new FileDataModel( new File( "data/merged.csv" ) );
						
			RecommenderIRStatsEvaluator evaluator = new GenericRecommenderIRStatsEvaluator();

			RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
				@Override
				public Recommender buildRecommender(DataModel model) throws TasteException {
					ItemSimilarity similarity = new PearsonCorrelationSimilarity(model);
					return new GenericItemBasedRecommender(model, similarity);
				} 
			};
			
			IRStatistics stats = evaluator.evaluate( recommenderBuilder, null, model, null, 5,
					GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1 );
			
			System.out.println( "Precision: " + stats.getPrecision() );
			System.out.println( "Recall: " + stats.getRecall() );
			System.out.println( "F1: " + stats.getF1Measure() );
			
			
		} catch (IOException e) {
			System.out.println("Error loading file");
			e.printStackTrace();
		} catch ( TasteException e ) {
			System.out.println("Error Mahout");
			e.printStackTrace();
		}
		
	}

}
