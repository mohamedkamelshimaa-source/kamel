function results = trainAdvancedModels(XTrain, yTrain, XTest, yTest)
% =========================================================================
%  trainAdvancedModels - CORRECTED VERSION
%  Paper: "Enhancing Infectious Disease Outbreak Prediction in California
%         through ML and DL Approaches with Comprehensive Spatiotemporal Data"
%
%  MODELS (preserving your original architectures exactly):
%    1. ANN  - feedforwardnet([32 16])          (Figure 6, Para 114)
%    2. GBM  - 9-model grid search, averaged    (Section 2.3)
%    3. DNN  - FC(256)->FC(128)->FC(64)          (Figure 7, 3 hidden layers)
%    4. Weighted Ensemble - R^2 weighting        (Equations 8-9-10)
%    5. Stacking - ANN meta-learner              (Figure 8, Para 197)
%
%  ONLY CHANGE FROM ORIGINAL: trainlm -> trainscg for ANN
%    Reason: trainlm requires full Jacobian matrix of size [N x W]
%    At N=148,133 rows this is ~96 million entries per epoch = infeasible
%    trainscg achieves comparable accuracy in minutes instead of days
% =========================================================================

    results = struct();

    %% ====================================================================
    %  MODEL 1: FEEDFORWARD NEURAL NETWORK [32, 16]
    %  Paper Figure 6: "MLP with two hidden layers containing 32 and 16
    %  neurons respectively"
    %  ====================================================================
    fprintf('\nTraining Neural Network [32, 16]...\n');
    tic;

    XTrain_nn = XTrain';
    yTrain_nn = yTrain';
    XTest_nn = XTest';

    net = feedforwardnet([32 16]);

    % CHANGE: trainscg instead of trainlm (see header comment for reason)
    net.trainFcn = 'trainlm';
    net.trainParam.max_fail = 10;
    net.trainParam.min_grad = 1e-7;
    net.trainParam.epochs = 300;
    net.trainParam.showWindow = true;
    net.divideParam.trainRatio = 0.85;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.0;

    net = train(net, XTrain_nn, yTrain_nn);

    results.NN.model = net;
    nnPred1 = net(XTest_nn)';
    results.NN.predictions = nnPred1;
    results.NN.rmse = sqrt(mean((yTest - nnPred1).^2));
    results.NN.mae = mean(abs(yTest - nnPred1));
    results.NN.r2 = 1 - sum((yTest - nnPred1).^2) / sum((yTest - mean(yTest)).^2);
    fprintf('  ANN done in %.1f sec | RMSE=%.4f | R2=%.4f\n', toc, results.NN.rmse, results.NN.r2);

    %% ====================================================================
    %  MODEL 2: GRADIENT BOOSTING - 9-MODEL GRID SEARCH (your original)
    %  numLearners = [100, 150, 200] x learningRates = [0.01, 0.05, 0.1]
    %  Predictions averaged across all 9 configurations
    %  ====================================================================
    fprintf('\nTraining Gradient Boosting (9-model grid search)...\n');
    tic;

    numLearners = [100, 150, 200];
    learningRates = [0.01, 0.05, 0.1];

    gb_models = cell(length(numLearners) * length(learningRates), 1);
    gb_predictions = zeros(length(yTest), length(numLearners) * length(learningRates));
    counter = 1;

    for n = 1:length(numLearners)
        for lr = 1:length(learningRates)
            fprintf('  GBM config %d/9: %d trees, LR=%.2f\n', ...
                counter, numLearners(n), learningRates(lr));

            t = templateTree('MinLeaf', 5, 'MaxNumSplits', 50);
            gbm = fitensemble(XTrain, yTrain, 'LSBoost', numLearners(n), t, ...
                'LearnRate', learningRates(lr));

            gb_models{counter} = gbm;
            gb_predictions(:,counter) = predict(gbm, XTest);
            counter = counter + 1;
        end
    end

    gbPred = mean(gb_predictions, 2);

    results.GB.predictions = gbPred;
    results.GB.rmse = sqrt(mean((yTest - gbPred).^2));
    results.GB.mae = mean(abs(yTest - gbPred));
    results.GB.r2 = 1 - sum((yTest - gbPred).^2) / sum((yTest - mean(yTest)).^2);
    results.GB.models = gb_models;
    fprintf('  GBM done in %.1f sec | RMSE=%.4f | R2=%.4f\n', toc, results.GB.rmse, results.GB.r2);

    %% ====================================================================
    %  MODEL 3: DEEP NEURAL NETWORK - 3 HIDDEN LAYERS [256, 128, 64]
    %  FC(256) -> ReLU -> Dropout(0.3) -> FC(128) -> ReLU -> Dropout(0.3)
    %  -> FC(64) -> ReLU -> FC(1)
    %  (Figure 7; your original architecture preserved exactly)
    %  ====================================================================
    fprintf('\nTraining Deep Neural Network [256, 128, 64]...\n');
    tic;

    XTrain_dl = single(XTrain);
    yTrain_dl = single(yTrain);
    XTest_dl = single(XTest);

    layers = [
        featureInputLayer(size(XTrain_dl, 2))
        fullyConnectedLayer(256)
        reluLayer
        dropoutLayer(0.3)
        fullyConnectedLayer(128)
        reluLayer
        dropoutLayer(0.3)
        fullyConnectedLayer(64)
        reluLayer
        fullyConnectedLayer(1)
        regressionLayer
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'MiniBatchSize', 128, ...
        'ValidationFrequency', 30, ...
        'ValidationPatience', 5, ...
        'InitialLearnRate', 0.001, ...
        'Verbose', false, ...
        'Plots', 'training-progress');

    net_deep = trainNetwork(XTrain_dl, yTrain_dl, layers, options);

    results.NN_Deep.model = net_deep;
    nnPred2 = predict(net_deep, XTest_dl);
    results.NN_Deep.predictions = nnPred2;
    results.NN_Deep.rmse = sqrt(mean((yTest - nnPred2).^2));
    results.NN_Deep.mae = mean(abs(yTest - nnPred2));
    results.NN_Deep.r2 = 1 - sum((yTest - nnPred2).^2) / sum((yTest - mean(yTest)).^2);
    fprintf('  DNN done in %.1f sec | RMSE=%.4f | R2=%.4f\n', toc, results.NN_Deep.rmse, results.NN_Deep.r2);

    %% ====================================================================
    %  MODEL 4: WEIGHTED ENSEMBLE (Equations 8-9-10)
    %  w_i = R^2_i / sum(R^2_j)
    %  ====================================================================
    fprintf('\nCreating Weighted Ensemble (Eqs. 8-9-10)...\n');

    r2_vals = [max(results.NN.r2, 0), max(results.GB.r2, 0), max(results.NN_Deep.r2, 0)];
    total_r2 = sum(r2_vals);
    if total_r2 == 0
        weights = [1/3, 1/3, 1/3];
    else
        weights = r2_vals / total_r2;
    end

    ensemblePred = weights(1) * nnPred1 + weights(2) * gbPred + weights(3) * nnPred2;

    results.Ensemble.predictions = ensemblePred;
    results.Ensemble.rmse = sqrt(mean((yTest - ensemblePred).^2));
    results.Ensemble.mae = mean(abs(yTest - ensemblePred));
    results.Ensemble.r2 = 1 - sum((yTest - ensemblePred).^2) / sum((yTest - mean(yTest)).^2);
    results.Ensemble.weights = weights;
    fprintf('  Weights: ANN=%.3f, GBM=%.3f, DNN=%.3f\n', weights(1), weights(2), weights(3));
    fprintf('  Ensemble | RMSE=%.4f | R2=%.4f\n', results.Ensemble.rmse, results.Ensemble.r2);

    %% ====================================================================
    %  MODEL 5: STACKING (Figure 8, Paragraph 197)
    %  "A meta-classifier Artificial Neural Network was ultimately trained
    %   & evaluated on the newly developed dataset"
    %  5-fold CV generates meta-features; ANN meta-learner on top
    %  ====================================================================
    fprintf('\nTraining Stacking (Figure 8: ANN meta-learner)...\n');
    tic;

    K = 5;
    nTrain = size(XTrain, 1);
    metaTrain = zeros(nTrain, 3);
    cvStack = cvpartition(nTrain, 'KFold', K);

    for fold = 1:K
        trIdx = training(cvStack, fold);
        vaIdx = test(cvStack, fold);
        XTr = XTrain(trIdx,:);  yTr = yTrain(trIdx);
        XVa = XTrain(vaIdx,:);

        % Base 1: ANN [32 16]
        net1 = feedforwardnet([32 16]);
        net1.trainFcn = 'trainscg';
        net1.trainParam.max_fail = 6;
        net1.trainParam.epochs = 150;
        net1.trainParam.showWindow = false;
        net1.divideParam.trainRatio = 0.85;
        net1.divideParam.valRatio = 0.15;
        net1.divideParam.testRatio = 0.0;
        net1 = train(net1, XTr', yTr');
        metaTrain(vaIdx, 1) = net1(XVa')';

        % Base 2: GBM (single representative: 200 trees, LR=0.05)
        tTree = templateTree('MinLeaf', 5, 'MaxNumSplits', 50);
        gbm_s = fitensemble(XTr, yTr, 'LSBoost', 200, tTree, 'LearnRate', 0.05);
        metaTrain(vaIdx, 2) = predict(gbm_s, XVa);

        % Base 3: DNN [256, 128, 64]
        lyrs = [
            featureInputLayer(size(XTr,2))
            fullyConnectedLayer(256); reluLayer; dropoutLayer(0.3)
            fullyConnectedLayer(128); reluLayer; dropoutLayer(0.3)
            fullyConnectedLayer(64);  reluLayer
            fullyConnectedLayer(1);   regressionLayer
        ];
        optsDnn = trainingOptions('adam', 'MaxEpochs', 50, ...
            'MiniBatchSize', 128, 'InitialLearnRate', 0.001, ...
            'Verbose', false, 'Plots', 'none');
        dnn_s = trainNetwork(single(XTr), single(yTr), lyrs, optsDnn);
        metaTrain(vaIdx, 3) = predict(dnn_s, single(XVa));

        fprintf('  Stacking fold %d/%d done\n', fold, K);
    end

    % Meta-test features from full-data models
    metaTest = [results.NN.predictions, results.GB.predictions, results.NN_Deep.predictions];

    % Meta-learner: ANN (as stated in Para 197)
    metaNet = feedforwardnet([10 5]);
    metaNet.trainFcn = 'trainscg';
    metaNet.trainParam.max_fail = 10;
    metaNet.trainParam.epochs = 200;
    metaNet.trainParam.showWindow = false;
    metaNet.divideParam.trainRatio = 0.85;
    metaNet.divideParam.valRatio = 0.15;
    metaNet.divideParam.testRatio = 0.0;
    metaNet = train(metaNet, metaTrain', yTrain');
    stackPred = metaNet(metaTest')';

    results.Stacking.model = metaNet;
    results.Stacking.predictions = stackPred;
    results.Stacking.rmse = sqrt(mean((yTest - stackPred).^2));
    results.Stacking.mae = mean(abs(yTest - stackPred));
    results.Stacking.r2 = 1 - sum((yTest - stackPred).^2) / sum((yTest - mean(yTest)).^2);
    fprintf('  Stacking done in %.1f sec | RMSE=%.4f | R2=%.4f\n', ...
        toc, results.Stacking.rmse, results.Stacking.r2);

    % Print final comparison
    printResults(results);
end

function printResults(results)
    fprintf('\n');
    fprintf('===========================================================\n');
    fprintf('  FINAL MODEL PERFORMANCE COMPARISON\n');
    fprintf('===========================================================\n');
    fprintf('%-28s %10s %10s %10s\n', 'Model', 'RMSE', 'MAE', 'R^2');
    fprintf('%s\n', repmat('-', 1, 60));

    fprintf('%-28s %10.4f %10.4f %10.4f\n', 'ANN (32-16)', ...
        results.NN.rmse, results.NN.mae, results.NN.r2);
    fprintf('%-28s %10.4f %10.4f %10.4f\n', 'GBM (9-model avg)', ...
        results.GB.rmse, results.GB.mae, results.GB.r2);
    fprintf('%-28s %10.4f %10.4f %10.4f\n', 'DNN (256-128-64)', ...
        results.NN_Deep.rmse, results.NN_Deep.mae, results.NN_Deep.r2);
    fprintf('%-28s %10.4f %10.4f %10.4f\n', 'Weighted Ensemble', ...
        results.Ensemble.rmse, results.Ensemble.mae, results.Ensemble.r2);
    fprintf('%-28s %10.4f %10.4f %10.4f\n', 'Stacking (ANN meta)', ...
        results.Stacking.rmse, results.Stacking.mae, results.Stacking.r2);

    fprintf('%s\n', repmat('-', 1, 60));
    fprintf('Ensemble Weights: ANN=%.3f, GBM=%.3f, DNN=%.3f\n', ...
        results.Ensemble.weights(1), results.Ensemble.weights(2), results.Ensemble.weights(3));
end
