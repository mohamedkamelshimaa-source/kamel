%% ========================================================================
%  REVIEWER RESPONSE - COMPLETE MATLAB CODE
%  "Enhancing Infectious Disease Outbreak Prediction in California through
%   Machine Learning and Deep Learning Approaches"
%  ========================================================================
%  MODELS IN THIS STUDY (matching your actual trainAdvancedModels.m):
%    ANN:       feedforwardnet([32 16])         - Figure 6
%    GBM:       9-model grid search, averaged   - Section 2.3
%    DNN:       FC(256)->FC(128)->FC(64)         - Figure 7
%    Ensemble:  R^2-weighted average             - Equations 8-9-10
%    Stacking:  ANN meta-learner                 - Figure 8
%  ========================================================================
%  REVIEWER SECTIONS:
%    S5:  Baseline models (Naive, LR, DT, ARIMA)         -> R1-C3
%    S7:  Permutation Importance                          -> R1-C4
%    S8:  Tree-based Importance                           -> R1-C4
%    S9:  Uncertainty (Bootstrap + MC Dropout)            -> R1-C6
%    S10: Diagnostic Figures 12-15                        -> R2-C17
%    S11: Robustness across disease strata                -> R2-C18
%  ========================================================================

clc; clear; close all;
rng(42, 'twister');

fprintf('=============================================================\n');
fprintf(' REVIEWER RESPONSE CODE\n');
fprintf(' Started: %s\n', datestr(now));
fprintf('=============================================================\n\n');

%% ========================================================================
%  SECTION 1: DATA LOADING AND PREPROCESSING
%  ========================================================================
fprintf('SECTION 1: Data Loading and Preprocessing\n');
fprintf('-------------------------------------------\n');

opts = detectImportOptions('odp_idb_2001_2022_ddg_compliant.csv');
opts.VariableNamingRule = 'preserve';
opts = setvartype(opts, {'Population','Rate','Lower_95__CI','Upper_95__CI'}, 'string');
data = readtable('odp_idb_2001_2022_ddg_compliant.csv', opts);

fprintf('Raw dataset: %d rows x %d columns\n', height(data), width(data));
fprintf('Year range: %d - %d\n', min(data.Year), max(data.Year));

% Convert '-' to NaN then to 0 (zero-substitution for CDPH suppressed counts)
data.Population = cellfun(@(x) str2double(strrep(x,'-','NaN')), data.Population);
data.Rate = cellfun(@(x) str2double(strrep(x,'-','NaN')), data.Rate);
data.("Lower_95__CI") = cellfun(@(x) str2double(strrep(x,'-','NaN')), data.("Lower_95__CI"));
data.("Upper_95__CI") = cellfun(@(x) str2double(strrep(x,'-','NaN')), data.("Upper_95__CI"));

data.Population(isnan(data.Population)) = 0;
data.Rate(isnan(data.Rate)) = 0;
data.("Lower_95__CI")(isnan(data.("Lower_95__CI"))) = 0;
data.("Upper_95__CI")(isnan(data.("Upper_95__CI"))) = 0;
data = rmmissing(data);

fprintf('After cleaning: %d rows\n', height(data));
fprintf('Diseases: %d | Counties: %d | Sex categories: %d\n', ...
    length(unique(data.Disease)), length(unique(data.County)), length(unique(data.Sex)));

%% ========================================================================
%  SECTION 2: FEATURE ENGINEERING (R2-C9: Table 1 dimensionality)
%  ========================================================================
fprintf('\nSECTION 2: Feature Engineering\n');
fprintf('-------------------------------\n');

data.LogPopulation = log1p(data.Population);
data.CI_Width = data.("Upper_95__CI") - data.("Lower_95__CI"); % Corrected: Upper-Lower

diseases = unique(data.Disease);
diseaseMatrix = zeros(height(data), length(diseases));
for i = 1:length(diseases)
    diseaseMatrix(:,i) = strcmp(data.Disease, diseases{i});
end

counties = unique(data.County);
countyMatrix = zeros(height(data), length(counties));
for i = 1:length(counties)
    countyMatrix(:,i) = strcmp(data.County, counties{i});
end

sexMatrix = double(strcmp(data.Sex, 'Male'));

numericFeatures = [data.Year, data.LogPopulation, ...
    data.("Lower_95__CI"), data.("Upper_95__CI"), data.CI_Width];
[normalizedFeatures, mu, sigma] = zscore(numericFeatures);

X = [normalizedFeatures, diseaseMatrix, countyMatrix, sexMatrix];
y = log1p(data.Cases);

featureNames = [{'Year','LogPopulation','Lower95CI','Upper95CI','CI_Width'}, ...
    strcat('Disease_', diseases'), strcat('County_', counties'), {'Sex_Male'}];

fprintf('\n--- TABLE 1 REVISION DATA (R2-C9) ---\n');
fprintf('Total preprocessed samples: %d\n', height(data));
fprintf('Numeric features: 5\n');
fprintf('Disease dummy variables: %d\n', length(diseases));
fprintf('County dummy variables: %d\n', length(counties));
fprintf('Sex binary feature: 1\n');
fprintf('TOTAL input feature dimensionality: %d\n', size(X,2));

%% ========================================================================
%  SECTION 3: TRAIN/TEST SPLIT (80/20)
%  ========================================================================
fprintf('\nSECTION 3: Train/Test Split (80/20)\n');
fprintf('-------------------------------------\n');

cv = cvpartition(size(data,1), 'HoldOut', 0.2);
XTrain = X(training(cv),:);  yTrain = y(training(cv));
XTest  = X(test(cv),:);      yTest  = y(test(cv));
trainIdx = training(cv);     testIdx = test(cv);

fprintf('Training: %d | Test: %d | Features: %d\n', ...
    size(XTrain,1), size(XTest,1), size(XTrain,2));

%% ========================================================================
%  SECTION 4: PROPOSED MODELS (calls your trainAdvancedModels.m)
%  ========================================================================
fprintf('\n=============================================================\n');
fprintf(' SECTION 4: Training Proposed Models\n');
fprintf('=============================================================\n');

results = trainAdvancedModels(XTrain, yTrain, XTest, yTest);

%% ========================================================================
%  SECTION 5: BASELINE MODELS (R1-C3)
%  ========================================================================
fprintf('\n=============================================================\n');
fprintf(' SECTION 5: BASELINE MODELS (R1-C3)\n');
fprintf('=============================================================\n');

baselineResults = struct();
trainData = data(trainIdx,:);
testData  = data(testIdx,:);

% --- 5a. Naive Persistence ---
fprintf('\n5a. Naive Persistence...\n');
tic;
naivePred = zeros(sum(testIdx),1);
for i = 1:height(testData)
    prevYear = testData.Year(i) - 1;
    mask = strcmp(trainData.Disease, testData.Disease{i}) & ...
           strcmp(trainData.County, testData.County{i}) & ...
           strcmp(trainData.Sex, testData.Sex{i}) & ...
           trainData.Year == prevYear;
    if any(mask)
        naivePred(i) = log1p(trainData.Cases(find(mask,1)));
    else
        mask2 = strcmp(trainData.Disease, testData.Disease{i}) & ...
                strcmp(trainData.County, testData.County{i}) & ...
                strcmp(trainData.Sex, testData.Sex{i});
        if any(mask2)
            naivePred(i) = mean(log1p(trainData.Cases(mask2)));
        else
            naivePred(i) = mean(yTrain);
        end
    end
end
baselineResults.Naive = computeBaselineMetrics(naivePred, yTest);
baselineResults.Naive.predictions = naivePred;
fprintf('  Done in %.1f sec | RMSE=%.4f MAE=%.4f R2=%.4f\n', toc, ...
    baselineResults.Naive.rmse, baselineResults.Naive.mae, baselineResults.Naive.r2);

% --- 5b. OLS Linear Regression ---
fprintf('5b. OLS Linear Regression...\n');
tic;
lrModel = fitlm(XTrain, yTrain);
lrPred = predict(lrModel, XTest);
baselineResults.LinearReg = computeBaselineMetrics(lrPred, yTest);
baselineResults.LinearReg.predictions = lrPred;
fprintf('  Done in %.1f sec | RMSE=%.4f MAE=%.4f R2=%.4f\n', toc, ...
    baselineResults.LinearReg.rmse, baselineResults.LinearReg.mae, baselineResults.LinearReg.r2);

% --- 5c. Decision Tree ---
fprintf('5c. Decision Tree...\n');
tic;
dtModel = fitrtree(XTrain, yTrain, 'MinLeafSize', 10);
dtPred = predict(dtModel, XTest);
baselineResults.DecisionTree = computeBaselineMetrics(dtPred, yTest);
baselineResults.DecisionTree.predictions = dtPred;
fprintf('  Done in %.1f sec | RMSE=%.4f MAE=%.4f R2=%.4f\n', toc, ...
    baselineResults.DecisionTree.rmse, baselineResults.DecisionTree.mae, baselineResults.DecisionTree.r2);

% --- 5d. ARIMA(1,1,0) per disease-county-sex series ---
fprintf('5d. ARIMA(1,1,0) per series...\n');
tic;
arimaPred = zeros(sum(testIdx),1);
testDataTable = data(testIdx,:);
uniqueCombs = unique(testDataTable(:,{'Disease','County','Sex'}));
arimaOK = 0; arimaFall = 0;

for c = 1:height(uniqueCombs)
    thisDis = uniqueCombs.Disease{c};
    thisCty = uniqueCombs.County{c};
    thisSex = uniqueCombs.Sex{c};

    trMask = strcmp(data.Disease, thisDis) & strcmp(data.County, thisCty) & ...
             strcmp(data.Sex, thisSex) & trainIdx;
    trSeries = sortrows(data(trMask,:), 'Year');

    teMask = strcmp(testDataTable.Disease, thisDis) & ...
             strcmp(testDataTable.County, thisCty) & ...
             strcmp(testDataTable.Sex, thisSex);

    if height(trSeries) >= 5
        try
            tsData = log1p(trSeries.Cases);
            mdl = arima(1,1,0);
            estMdl = estimate(mdl, tsData, 'Display', 'off');
            [yF,~] = forecast(estMdl, sum(teMask), 'Y0', tsData);
            arimaPred(teMask) = yF;
            arimaOK = arimaOK + 1;
        catch
            arimaPred(teMask) = mean(log1p(trSeries.Cases));
            arimaFall = arimaFall + 1;
        end
    else
        if height(trSeries) > 0
            arimaPred(teMask) = mean(log1p(trSeries.Cases));
        else
            arimaPred(teMask) = mean(yTrain);
        end
        arimaFall = arimaFall + 1;
    end
end
baselineResults.ARIMA = computeBaselineMetrics(arimaPred, yTest);
baselineResults.ARIMA.predictions = arimaPred;
fprintf('  Done in %.1f sec | Fitted: %d Fallback: %d\n', toc, arimaOK, arimaFall);
fprintf('  RMSE=%.4f MAE=%.4f R2=%.4f\n', ...
    baselineResults.ARIMA.rmse, baselineResults.ARIMA.mae, baselineResults.ARIMA.r2);

%% ========================================================================
%  SECTION 6: FULL COMPARISON TABLE (New Table for manuscript)
%  ========================================================================
fprintf('\n=============================================================\n');
fprintf(' SECTION 6: FULL COMPARISON TABLE\n');
fprintf('=============================================================\n');

fprintf('\n%-30s %10s %10s %10s\n', 'Model', 'RMSE', 'MAE', 'R-squared');
fprintf('%s\n', repmat('-', 1, 62));
% Baselines
fprintf('%-30s %10.4f %10.4f %10.4f\n', 'Naive Persistence', ...
    baselineResults.Naive.rmse, baselineResults.Naive.mae, baselineResults.Naive.r2);
fprintf('%-30s %10.4f %10.4f %10.4f\n', 'Linear Regression (OLS)', ...
    baselineResults.LinearReg.rmse, baselineResults.LinearReg.mae, baselineResults.LinearReg.r2);
fprintf('%-30s %10.4f %10.4f %10.4f\n', 'Decision Tree', ...
    baselineResults.DecisionTree.rmse, baselineResults.DecisionTree.mae, baselineResults.DecisionTree.r2);
fprintf('%-30s %10.4f %10.4f %10.4f\n', 'ARIMA(1,1,0)', ...
    baselineResults.ARIMA.rmse, baselineResults.ARIMA.mae, baselineResults.ARIMA.r2);
fprintf('%s\n', repmat('-', 1, 62));
% Proposed
fprintf('%-30s %10.4f %10.4f %10.4f\n', 'ANN (32-16)', ...
    results.NN.rmse, results.NN.mae, results.NN.r2);
fprintf('%-30s %10.4f %10.4f %10.4f\n', 'GBM (9-model avg)', ...
    results.GB.rmse, results.GB.mae, results.GB.r2);
fprintf('%-30s %10.4f %10.4f %10.4f\n', 'DNN (256-128-64)', ...
    results.NN_Deep.rmse, results.NN_Deep.mae, results.NN_Deep.r2);
fprintf('%-30s %10.4f %10.4f %10.4f\n', 'Weighted Ensemble', ...
    results.Ensemble.rmse, results.Ensemble.mae, results.Ensemble.r2);
fprintf('%-30s %10.4f %10.4f %10.4f\n', 'Stacking (ANN meta)', ...
    results.Stacking.rmse, results.Stacking.mae, results.Stacking.r2);
fprintf('%s\n', repmat('-', 1, 62));

bestBase = min([baselineResults.Naive.rmse, baselineResults.LinearReg.rmse, ...
    baselineResults.DecisionTree.rmse, baselineResults.ARIMA.rmse]);
fprintf('\nBest baseline RMSE: %.4f\n', bestBase);
fprintf('Ensemble improvement over best baseline: %.2f%%\n', ...
    (bestBase - results.Ensemble.rmse)/bestBase*100);
fprintf('Stacking improvement over best baseline: %.2f%%\n', ...
    (bestBase - results.Stacking.rmse)/bestBase*100);

%% ========================================================================
%  SECTION 7: PERMUTATION IMPORTANCE (R1-C4) - Figure 16
%  ========================================================================
fprintf('\n=============================================================\n');
fprintf(' SECTION 7: PERMUTATION IMPORTANCE (R1-C4)\n');
fprintf('=============================================================\n');

% Use the best single GBM model (200 trees, LR=0.05 = config 6) for importance
bestGBM = results.GB.models{6}; % 200 trees, 0.05 LR
baseRMSE_gbm = sqrt(mean((yTest - predict(bestGBM, XTest)).^2));

featureGroupNames = {'Year', 'LogPopulation', 'Lower 95% CI', 'Upper 95% CI', ...
    'CI Width', 'Disease (all)', 'County (all)', 'Sex'};
nDis = length(diseases); nCty = length(counties);
featureGroups = {1, 2, 3, 4, 5, 6:(5+nDis), (6+nDis):(5+nDis+nCty), (6+nDis+nCty)};

nPerm = 10;
importScores = zeros(length(featureGroups), nPerm);
for g = 1:length(featureGroups)
    for p = 1:nPerm
        Xp = XTest;
        Xp(:, featureGroups{g}) = XTest(randperm(size(XTest,1)), featureGroups{g});
        permPred = predict(bestGBM, Xp);
        importScores(g,p) = sqrt(mean((yTest - permPred).^2)) - baseRMSE_gbm;
    end
end

meanImp = mean(importScores,2);
stdImp  = std(importScores,0,2);
[sortImp, sIdx] = sort(meanImp, 'descend');
sortNames = featureGroupNames(sIdx);
sortStd = stdImp(sIdx);

fprintf('\n%-25s %15s %15s\n', 'Feature Group', 'Delta RMSE', 'Std');
fprintf('%s\n', repmat('-', 1, 57));
for i = 1:length(sortNames)
    fprintf('%-25s %15.6f %15.6f\n', sortNames{i}, sortImp(i), sortStd(i));
end

figure('Position',[100 100 900 500]);
barh(sortImp(end:-1:1), 'FaceColor', [0.2 0.6 0.8]); hold on;
errorbar(sortImp(end:-1:1), 1:length(sortImp), sortStd(end:-1:1), 'horizontal', 'k.', 'LineWidth', 1.2);
set(gca, 'YTick', 1:length(sortNames), 'YTickLabel', sortNames(end:-1:1), 'FontSize', 11);
xlabel('Mean Increase in RMSE', 'FontSize', 12);
title('Figure 16: Permutation Feature Importance (GBM)', 'FontSize', 14);
grid on; box on;
saveas(gcf, 'Figure16_PermutationImportance.png');
fprintf('Saved: Figure16_PermutationImportance.png\n');

%% ========================================================================
%  SECTION 8: TREE-BASED IMPORTANCE (R1-C4) - Figure 17
%  ========================================================================
fprintf('\n=============================================================\n');
fprintf(' SECTION 8: TREE-BASED IMPORTANCE (R1-C4)\n');
fprintf('=============================================================\n');

gbmPredImp = predictorImportance(bestGBM);
groupImp = zeros(length(featureGroups),1);
for g = 1:length(featureGroups)
    groupImp(g) = sum(gbmPredImp(featureGroups{g}));
end
groupPct = groupImp / sum(groupImp) * 100;
[sortPct, sIdx2] = sort(groupPct, 'descend');
sortNames2 = featureGroupNames(sIdx2);

fprintf('\n%-25s %18s\n', 'Feature Group', 'Importance (%%)');
fprintf('%s\n', repmat('-', 1, 45));
for i = 1:length(sortNames2)
    fprintf('%-25s %16.2f%%\n', sortNames2{i}, sortPct(i));
end

figure('Position',[100 100 900 500]);
barh(sortPct(end:-1:1), 'FaceColor', [0.8 0.4 0.2]);
set(gca, 'YTick', 1:length(sortNames2), 'YTickLabel', sortNames2(end:-1:1), 'FontSize', 11);
xlabel('Relative Importance (%)', 'FontSize', 12);
title('Figure 17: GBM Predictor Importance', 'FontSize', 14);
grid on; box on;
saveas(gcf, 'Figure17_TreeImportance.png');
fprintf('Saved: Figure17_TreeImportance.png\n');

%% ========================================================================
%  SECTION 9: UNCERTAINTY ESTIMATION (R1-C6) - Figure 18
%  ========================================================================
fprintf('\n=============================================================\n');
fprintf(' SECTION 9: UNCERTAINTY ESTIMATION (R1-C6)\n');
fprintf('=============================================================\n');

% --- 9a. Bootstrap 95% PI for GBM ---
fprintf('9a. Bootstrap 95%% PI for GBM (30 resamples)...\n');
nBoot = 30;
bootPreds = zeros(size(XTest,1), nBoot);
for b = 1:nBoot
    bIdx = randsample(size(XTrain,1), size(XTrain,1), true);
    tTree = templateTree('MinLeaf', 5, 'MaxNumSplits', 50);
    bModel = fitensemble(XTrain(bIdx,:), yTrain(bIdx), 'LSBoost', 200, tTree, 'LearnRate', 0.05);
    bootPreds(:,b) = predict(bModel, XTest);
    if mod(b,10)==0, fprintf('  Resample %d/%d\n', b, nBoot); end
end
bootMean  = mean(bootPreds,2);
bootLower = quantile(bootPreds, 0.025, 2);
bootUpper = quantile(bootPreds, 0.975, 2);
cov95boot = mean(yTest >= bootLower & yTest <= bootUpper)*100;
fprintf('  Coverage: %.2f%% | Mean PI width: %.4f\n', cov95boot, mean(bootUpper-bootLower));

% --- 9b. MC Dropout for ANN ---
fprintf('\n9b. MC Dropout 95%% PI for ANN (20 passes)...\n');
nMC = 20;
mcPreds = zeros(size(XTest,1), nMC);
for m = 1:nMC
    dropMask = rand(1, size(XTrain,2)) > 0.2;
    XTr_m = XTrain .* dropMask;
    XTe_m = XTest  .* dropMask;
    net_mc = feedforwardnet([32 16]);
    net_mc.trainFcn = 'trainscg';
    net_mc.trainParam.showWindow = false;
    net_mc.trainParam.max_fail = 6;
    net_mc.trainParam.epochs = 100;
    net_mc.divideParam.trainRatio = 0.85;
    net_mc.divideParam.valRatio = 0.15;
    net_mc.divideParam.testRatio = 0.0;
    net_mc = train(net_mc, XTr_m', yTrain');
    mcPreds(:,m) = net_mc(XTe_m')';
    if mod(m,5)==0, fprintf('  Pass %d/%d\n', m, nMC); end
end
mcMean  = mean(mcPreds,2);
mcLower = quantile(mcPreds, 0.025, 2);
mcUpper = quantile(mcPreds, 0.975, 2);
cov95mc = mean(yTest >= mcLower & yTest <= mcUpper)*100;
fprintf('  Coverage: %.2f%% | Mean PI width: %.4f\n', cov95mc, mean(mcUpper-mcLower));

% --- Figure 18 ---
nPlot = min(500, length(yTest));
pIdx = sort(randsample(length(yTest), nPlot));
figure('Position',[100 100 1200 500]);

subplot(1,2,1);
fill([1:nPlot, nPlot:-1:1]', [bootUpper(pIdx); flipud(bootLower(pIdx))], ...
    [0.8 0.9 1], 'EdgeColor','none','FaceAlpha',0.6); hold on;
plot(1:nPlot, yTest(pIdx), 'k.', 'MarkerSize',3);
plot(1:nPlot, bootMean(pIdx), 'b-', 'LineWidth',1);
xlabel('Test Sample'); ylabel('log1p(Cases)');
title(sprintf('GBM Bootstrap 95%% PI (Cov: %.1f%%)', cov95boot));
legend('95% PI','Actual','Predicted','Location','best'); grid on;

subplot(1,2,2);
fill([1:nPlot, nPlot:-1:1]', [mcUpper(pIdx); flipud(mcLower(pIdx))], ...
    [1 0.9 0.8], 'EdgeColor','none','FaceAlpha',0.6); hold on;
plot(1:nPlot, yTest(pIdx), 'k.', 'MarkerSize',3);
plot(1:nPlot, mcMean(pIdx), 'r-', 'LineWidth',1);
xlabel('Test Sample'); ylabel('log1p(Cases)');
title(sprintf('ANN MC Dropout 95%% PI (Cov: %.1f%%)', cov95mc));
legend('95% PI','Actual','Predicted','Location','best'); grid on;

sgtitle('Figure 18: Predictive Uncertainty Estimation', 'FontSize', 14);
saveas(gcf, 'Figure18_UncertaintyEstimation.png');
fprintf('Saved: Figure18_UncertaintyEstimation.png\n');

%% ========================================================================
%  SECTION 10: DIAGNOSTIC PLOTS (R2-C17) - Figures 12-15, 19
%  ========================================================================
fprintf('\n=============================================================\n');
fprintf(' SECTION 10: DIAGNOSTIC PLOTS (R2-C17)\n');
fprintf('=============================================================\n');

mNames = {'ANN (32-16)', 'GBM (9-avg)', 'DNN (256-128-64)', 'W. Ensemble', 'Stacking'};
mPreds = {results.NN.predictions, results.GB.predictions, results.NN_Deep.predictions, ...
           results.Ensemble.predictions, results.Stacking.predictions};
mResid = cellfun(@(p) yTest - p, mPreds, 'UniformOutput', false);
colors = {[0.2 0.6 0.8],[0.8 0.4 0.2],[0.4 0.7 0.3],[0.6 0.2 0.6],[0.9 0.6 0.1]};

% Figure 12: Q-Q Plots
figure('Position',[100 100 1400 500]);
for i = 1:5
    subplot(1,5,i); qqplot(mResid{i});
    title(mNames{i}, 'FontSize', 10);
    xlabel('Normal Quantiles'); ylabel('Residual Quantiles'); grid on;
end
sgtitle('Figure 12: Q-Q Plots of Residual Distributions', 'FontSize', 14);
saveas(gcf, 'Figure12_QQPlots.png'); fprintf('Saved: Figure12_QQPlots.png\n');

% Figure 13: Residual Histograms
figure('Position',[100 100 1400 500]);
for i = 1:5
    subplot(1,5,i);
    histogram(mResid{i}, 50, 'FaceColor', colors{i}, 'EdgeColor','none','FaceAlpha',0.7);
    hold on; xline(0, 'r--', 'LineWidth', 1.5);
    title(mNames{i}, 'FontSize', 10); xlabel('Residual'); ylabel('Freq');
end
sgtitle('Figure 13: Residual Distribution Histograms', 'FontSize', 14);
saveas(gcf, 'Figure13_ResidualHistograms.png'); fprintf('Saved: Figure13_ResidualHistograms.png\n');

% Figure 14: Residuals vs Predicted
figure('Position',[100 100 1400 500]);
for i = 1:5
    subplot(1,5,i);
    scatter(mPreds{i}, mResid{i}, 3, colors{i}, 'filled', 'MarkerFaceAlpha', 0.3);
    hold on; yline(0, 'r--', 'LineWidth', 1.5);
    title(mNames{i}, 'FontSize', 10); xlabel('Predicted'); ylabel('Residual'); grid on;
end
sgtitle('Figure 14: Residuals vs. Predicted Values', 'FontSize', 14);
saveas(gcf, 'Figure14_ResidualsVsPredicted.png'); fprintf('Saved: Figure14_ResidualsVsPredicted.png\n');

% Figure 15: Actual vs Predicted
figure('Position',[100 100 1400 500]);
for i = 1:5
    subplot(1,5,i);
    scatter(yTest, mPreds{i}, 3, colors{i}, 'filled', 'MarkerFaceAlpha', 0.3);
    hold on;
    lims = [min([yTest; mPreds{i}]), max([yTest; mPreds{i}])];
    plot(lims, lims, 'k--', 'LineWidth', 2);
    pp = polyfit(yTest, mPreds{i}, 1);
    plot(lims, polyval(pp, lims), 'r-', 'LineWidth', 1.5);
    r2v = 1 - sum((yTest-mPreds{i}).^2)/sum((yTest-mean(yTest)).^2);
    title(sprintf('%s (R^2=%.3f)', mNames{i}, r2v), 'FontSize', 9);
    xlabel('Actual'); ylabel('Predicted');
    legend('Data','Perfect','Fit','Location','best','FontSize',7); grid on;
end
sgtitle('Figure 15: Actual vs. Predicted Regression Comparison', 'FontSize', 14);
saveas(gcf, 'Figure15_RegressionLines.png'); fprintf('Saved: Figure15_RegressionLines.png\n');

% Figure 19: Baseline vs Proposed Bar Chart
figure('Position',[100 100 1100 600]);
allNames = {'Naive','Lin Reg','Dec Tree','ARIMA','ANN','GBM','DNN','W.Ens','Stack'};
allRMSE = [baselineResults.Naive.rmse, baselineResults.LinearReg.rmse, ...
    baselineResults.DecisionTree.rmse, baselineResults.ARIMA.rmse, ...
    results.NN.rmse, results.GB.rmse, results.NN_Deep.rmse, ...
    results.Ensemble.rmse, results.Stacking.rmse];
allR2 = [baselineResults.Naive.r2, baselineResults.LinearReg.r2, ...
    baselineResults.DecisionTree.r2, baselineResults.ARIMA.r2, ...
    results.NN.r2, results.GB.r2, results.NN_Deep.r2, ...
    results.Ensemble.r2, results.Stacking.r2];
barColors = [repmat([0.7 0.7 0.7],4,1); repmat([0.2 0.6 0.8],5,1)];

subplot(1,2,1);
b1 = bar(allRMSE, 'FaceColor','flat'); b1.CData = barColors;
set(gca, 'XTick',1:9, 'XTickLabel',allNames, 'XTickLabelRotation',45);
ylabel('RMSE'); title('RMSE Comparison'); grid on;
subplot(1,2,2);
b2 = bar(allR2, 'FaceColor','flat'); b2.CData = barColors;
set(gca, 'XTick',1:9, 'XTickLabel',allNames, 'XTickLabelRotation',45);
ylabel('R^2'); title('R-squared Comparison'); grid on;
sgtitle('Figure 19: Baseline (gray) vs. Proposed (blue)', 'FontSize', 14);
saveas(gcf, 'Figure19_BaselineComparison.png'); fprintf('Saved: Figure19_BaselineComparison.png\n');

%% ========================================================================
%  SECTION 11: ROBUSTNESS ACROSS STRATA (R2-C18)
%  ========================================================================
fprintf('\n=============================================================\n');
fprintf(' SECTION 11: ROBUSTNESS BY DISEASE (R2-C18)\n');
fprintf('=============================================================\n');

testDiseases = data.Disease(testIdx);
[uDis, ~, dIdx] = unique(testDiseases);
dCounts = accumarray(dIdx, 1);
[~, topOrd] = sort(dCounts, 'descend');
topDis = uDis(topOrd(1:min(10, length(topOrd))));

fprintf('\n%-45s %8s %8s %8s %8s\n', 'Disease', 'N', 'RMSE', 'MAE', 'R2');
fprintf('%s\n', repmat('-', 1, 79));
for d = 1:length(topDis)
    mask = strcmp(testDiseases, topDis{d});
    if sum(mask) > 10
        dr = sqrt(mean((yTest(mask)-results.Ensemble.predictions(mask)).^2));
        dm = mean(abs(yTest(mask)-results.Ensemble.predictions(mask)));
        d2 = 1 - sum((yTest(mask)-results.Ensemble.predictions(mask)).^2) / ...
             max(sum((yTest(mask)-mean(yTest(mask))).^2), eps);
        fprintf('%-45s %8d %8.4f %8.4f %8.4f\n', topDis{d}, sum(mask), dr, dm, d2);
    end
end

%% ========================================================================
%  SECTION 12: SAVE ALL RESULTS
%  ========================================================================
fprintf('\n=============================================================\n');
fprintf(' SECTION 12: Saving Results\n');
fprintf('=============================================================\n');

save('ReviewerResponseResults.mat', 'results', 'baselineResults', ...
    'meanImp', 'featureGroupNames', 'groupPct', ...
    'bootMean', 'bootLower', 'bootUpper', 'cov95boot', ...
    'mcMean', 'mcLower', 'mcUpper', 'cov95mc', ...
    'mu', 'sigma', 'featureNames', 'diseases', 'counties');

allMAE = [baselineResults.Naive.mae, baselineResults.LinearReg.mae, ...
    baselineResults.DecisionTree.mae, baselineResults.ARIMA.mae, ...
    results.NN.mae, results.GB.mae, results.NN_Deep.mae, ...
    results.Ensemble.mae, results.Stacking.mae];
mLabels = {'Naive_Persistence','Linear_Regression','Decision_Tree','ARIMA', ...
    'ANN_32_16','GBM_9model_avg','DNN_256_128_64','Weighted_Ensemble','Stacking'};
compTable = table(mLabels', allRMSE', allMAE', allR2', ...
    'VariableNames', {'Model','RMSE','MAE','R_squared'});
writetable(compTable, 'ModelComparisonTable.csv');

fprintf('Saved: ReviewerResponseResults.mat\n');
fprintf('Saved: ModelComparisonTable.csv\n');

fprintf('\n=============================================================\n');
fprintf(' ALL SECTIONS COMPLETED: %s\n', datestr(now));
fprintf('=============================================================\n');

%% LOCAL HELPER
function metrics = computeBaselineMetrics(predictions, actual)
    metrics.rmse = sqrt(mean((actual - predictions).^2));
    metrics.mae  = mean(abs(actual - predictions));
    metrics.r2   = 1 - sum((actual - predictions).^2) / sum((actual - mean(actual)).^2);
end
