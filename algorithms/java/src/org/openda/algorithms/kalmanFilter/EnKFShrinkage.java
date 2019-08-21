package org.openda.algorithms.kalmanFilter;

import org.openda.interfaces.*;
import org.openda.utils.Matrix;
import org.openda.utils.Results;

import java.io.File;

public class EnKFShrinkage extends EnKF {

	protected enum SKMethodType{none,RBLW,OAS, LW, DS}
	protected SKMethodType SKMethod=SKMethodType.none;

	public void initialize(File workingDir, String[] arguments) {

		super.initialize(workingDir, arguments);

		String sK = this.configurationAsTree.getAsString("covarianceEstimator","traditional");
		System.out.println("Selected Shrinkage Estimator method:"+sK);

		if ( sK.equalsIgnoreCase("traditional") ) {
			this.SKMethod = SKMethodType.none;
		}
		else if (sK.equalsIgnoreCase("rblw")){
			this.SKMethod = SKMethodType.RBLW;
		}
		else if (sK.equalsIgnoreCase("oas")){
			this.SKMethod = SKMethodType.OAS;
		}
		else if (sK.equalsIgnoreCase("lw")){
			this.SKMethod = SKMethodType.LW;
		}
		else if (sK.equalsIgnoreCase("ds")){
			this.SKMethod = SKMethodType.DS;
		}
		else {
			throw new RuntimeException("Shrinkage covariance estimator type '" + sK +  "' is not supported");
		}
	}

	protected void applyShrinkageToCovariance(EnsembleVectors ensembleVectors, Matrix predMat, Matrix D , double [] alphaU , int n, int q, int m, double sqrtQmin1){
    	/*
    	Applying methods proposed by Elias Nino and Luis Guzman.
    	 */
		Matrix delta =  (new Matrix(ensembleVectors.ensemble));
		delta.scale(1/sqrtQmin1);
		Matrix[] SVD = delta.svd();
		double S2 = 0.0;
		double S = 0.0;
		for	(int i=0;i< Math.min(SVD[1].getNumberOfRows(),SVD[1].getNumberOfColumns());i++) {
			double tempS =  SVD[1].getValue(i, i)* SVD[1].getValue(i, i);
			S = S + tempS;
			tempS =  tempS* tempS;
			S2 = S2 + tempS;
		}

		alphaU[1] = Math.max(0.15,(1+S)/n);

		switch (this.SKMethod){
			case OAS:{
				alphaU[0] = Math.min(((1 - 2 / n) * S2 + S * S) / ((q + 1 - 2 / n) * (S2 - (S * S) / n)), 1);
				break;
			}
			case LW:{
				double suma=0;
				for (int k=0;k<q;k++){
					double norm=0;
					for (int j=0;j<n;j++){
						norm=norm + predMat.getValue(j,k)*predMat.getValue(j,k);
					}
					suma=suma + norm*norm;
				}
				alphaU[0] = Math.min((suma-(q-2)*S2)/((S2-(S*S)/n)*q*q),1);
				break;
			}
			case RBLW:{
				alphaU[0] = Math.min((((q-2)/q)*S2+ S*S)/((q+2)*(S2-(S*S)/n)), 1);
				break;
			}
			case DS: {
				double u = S/n;
				int cont = 0;
				while (cont <= q &&  SVD[1].getValue(cont, cont) > u ){
					cont = cont+1;
				}
				if (cont/n < 0.2){
					alphaU[0] = Math.min(((1 - 2 / n) * S2 + S * S) / ((q + 1 - 2 / n) * (S2 - (S * S) / n)), 1);
				}else{
					alphaU[0] = Math.min((((q-2)/q)*S2+ S*S)/((q+2)*(S2-(S*S)/n)), 1);
				}

				break;
			}
		}
		predMat.scale(Math.sqrt((1.0-alphaU[0]) / (q - 1.0)));
		D.multiply(1.0,  predMat  , predMat, 0.0, false, true);
		for	(int i=0;i<m;i++) {
			D.setValue(i, i, D.getValue(i, i)+ alphaU[0]*alphaU[1]);
		}
	}

	protected  void updateVectorKGainShrinkage(IStochObserver obs, IVector[] Kvecs, Matrix inverseD  , double [] alphaU , int i, int m ){
		IObservationDescriptions descr = obs.getObservationDescriptions() ;
		IVector obsIndex = descr.getValueProperties("index");
		if(obsIndex==null){
			obsIndex = descr.getValueProperties("xPosition");
		}
		for(int j=0;j<m;j++) {
			int indx = (int) obsIndex.getValue(j);
			Kvecs[i].setValue(indx, Kvecs[i].getValue(indx) + alphaU[0] * alphaU[1] * inverseD.getValue(j, i));
		}
	}

	@Override
	protected IVector[] computeGainMatrix(IStochObserver obs, EnsembleVectors ensemblePredictions, EnsembleVectors ensembleVectors, boolean compute_pred_a_linear, boolean write_output){
		int m = obs.getCount(); // number of observations
		int n = ensembleVectors.mean.getSize(); // length of the state vector
		int q = this.ensembleSize; // number of ensemble members
		double sqrtQmin1 = Math.sqrt((double) q -1.0);


		// compute Kalman gain
		// D = HPH+R = (1/(q-1))PRED*PRED'+sqrtR*sqrtR' : covariance of
		// innovations
		timerLinalg.start();
		//H*A^f_k = predMat = prediction of the observed model values, after removing the mean.
		Matrix predMat = new Matrix(ensemblePredictions.ensemble);
		Matrix D = new Matrix(m, m);

		//Apply shrinkage
		double alphaU [] =  new double[2];
		if (this.SKMethod != SKMethodType.none) {
			applyShrinkageToCovariance(ensembleVectors,predMat, D , alphaU,n, q, m, sqrtQmin1);
		}else {
			predMat.scale(Math.sqrt(1.0 / (q - 1.0)));
			D.multiply(1.0,  predMat  , predMat, 0.0, false, true);
		}

		// System.out.println("predMat="+predMat);


		IMatrix sqrtR = obs.getSqrtCovariance().asMatrix();
		D.multiply(1.0, sqrtR, sqrtR, 1.0, false, true);
		timerLinalg.stop();
		timerResults.start();
		if (write_output){
			Results.putValue("sqrt_r", sqrtR, sqrtR.getNumberOfColumns() * sqrtR.getNumberOfRows() , "analysis step", IResultWriter.OutputLevel.All, IResultWriter.MessageType.Step);
			Results.putValue("hpht_plus_r", D, D.getNumberOfColumns() * D.getNumberOfRows() , "analysis step", IResultWriter.OutputLevel.All, IResultWriter.MessageType.Step);
			Results.putProgression("length of state vector: " + n + ".");
			Results.putProgression("number of observations: " + m + ".");
		}
		timerResults.stop();
		timerLinalg.start();
		// K = XI*PRED'*inv(D) = XI * Xfac
		for(int i=0;i<q;i++){
			ensembleVectors.ensemble[i].scale(Math.sqrt(1-alphaU[0]) /sqrtQmin1);
		}

		// System.out.println("PHT="+K);
		Matrix inverseD = D.inverse();
		// System.out.println("inverseD="+inverseD);

		// version without large matrices
		// K = XI * E with E=PRED'*inv(D)
		Matrix E = new Matrix(q,m);
		E.multiply(1.0, predMat, inverseD, 0.0, true, false);
		IVector Kvecs[] = new IVector[m];
		timerLinalg.stop();
		for(int i=0;i<m;i++){
			timerLinalg.start();
			Kvecs[i] = ensembleVectors.ensemble[0].clone(); //HERE !!!
			Kvecs[i].scale(0.0);
			for(int j=0;j<q;j++){
				Kvecs[i].axpy(E.getValue(j, i),ensembleVectors.ensemble[j]);
			}
			if (this.SKMethod!=SKMethodType.none){
				updateVectorKGainShrinkage(obs, Kvecs, inverseD  ,alphaU , i,  m);
			}

			timerLinalg.stop();
			timerResults.start();
			if (write_output){
				Results.putValue("k_"+i, Kvecs[i], Kvecs[i].getSize() , "analysis step", IResultWriter.OutputLevel.All, IResultWriter.MessageType.Step);
			}
			timerResults.stop();
		}

		for(int i=0;i<q;i++){
			ensembleVectors.ensemble[i].scale(sqrtQmin1/Math.sqrt(1-alphaU[0]));
		}

		if (compute_pred_a_linear){
			// Compute H*K for linear update of predictions, since for blackbox models the predictions
			// are not upadted until after the next forecast
			// H*K = PRED*PRED'*inv(D)
			timerLinalg.start();
			Matrix K_pred = new Matrix(m,m);
			K_pred.multiply(1.0, predMat, E, 0.0, false, false);
			// pred_a_linear = predAvg + K_pred*(obsVal-predAvg)
			IVector innovAvg = obs.getExpectations();
			innovAvg.axpy(-1, ensemblePredictions.mean);
			IVector pred_a_linear = ensemblePredictions.mean.clone();
			K_pred.rightMultiply(1.0, innovAvg, 1.0, pred_a_linear);
			innovAvg.free();
			timerLinalg.stop();
			timerResults.start();
			if (write_output){
				Results.putValue("pred_a_linear", pred_a_linear, pred_a_linear.getSize() , "analysis step", IResultWriter.OutputLevel.Normal, IResultWriter.MessageType.Step);
			}
			timerResults.stop();

			pred_a_linear.free();
		}

		return Kvecs;
	}


}
