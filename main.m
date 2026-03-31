% =========================================================================
%  kokoko.m - Data Preprocessing (CORRECTED)
%  Fix: CI_Width = Upper - Lower (original had Lower - Upper = negative)
% =========================================================================

% Load data with preserved column names
opts = detectImportOptions('odp_idb_2001_2022_ddg_compliant.csv');
opts.VariableNamingRule = 'preserve';
opts = setvartype(opts, {'Population', 'Rate', 'Lower_95__CI', 'Upper_95__CI'}, 'string');
data = readtable('odp_idb_2001_2022_ddg_compliant.csv', opts);

% Convert '-' to NaN then to 0 (CDPH suppressed counts treated as zero)
data.Population = cellfun(@(x) str2double(strrep(x, '-', 'NaN')), data.Population);
data.Rate = cellfun(@(x) str2double(strrep(x, '-', 'NaN')), data.Rate);
data.("Lower_95__CI") = cellfun(@(x) str2double(strrep(x, '-', 'NaN')), data.("Lower_95__CI"));
data.("Upper_95__CI") = cellfun(@(x) str2double(strrep(x, '-', 'NaN')), data.("Upper_95__CI"));

data.Population(isnan(data.Population)) = 0;
data.Rate(isnan(data.Rate)) = 0;
data.("Lower_95__CI")(isnan(data.("Lower_95__CI"))) = 0;
data.("Upper_95__CI")(isnan(data.("Upper_95__CI"))) = 0;
data = rmmissing(data);

% Feature engineering
data.LogPopulation = log1p(data.Population);

% CORRECTED: Upper - Lower (was Lower - Upper in original)
data.CI_Width = data.("Upper_95__CI") - data.("Lower_95__CI");

% Disease encoding (one-hot)
diseases = unique(data.Disease);
diseaseMatrix = zeros(height(data), length(diseases));
for i = 1:length(diseases)
    diseaseMatrix(:,i) = strcmp(data.Disease, diseases{i});
end

% County encoding (one-hot)
counties = unique(data.County);
countyMatrix = zeros(height(data), length(counties));
for i = 1:length(counties)
    countyMatrix(:,i) = strcmp(data.County, counties{i});
end

% Sex encoding (binary: 1 = Male, 0 = Female/Total)
sexMatrix = double(strcmp(data.Sex, 'Male'));

% Numeric features
numericFeatures = [data.Year, data.LogPopulation, ...
    data.("Lower_95__CI"), data.("Upper_95__CI"), data.CI_Width];

% Z-score normalization
[normalizedFeatures, mu, sigma] = zscore(numericFeatures);

% Combine all features
X = [normalizedFeatures, diseaseMatrix, countyMatrix, sexMatrix];

% Target: log1p(Cases)
y = log1p(data.Cases);

% Split data (80/20)
rng(42, 'twister');
cv = cvpartition(size(data,1), 'HoldOut', 0.2);
XTrain = X(training(cv),:);
yTrain = y(training(cv),:);
XTest = X(test(cv),:);
yTest = y(test(cv),:);

fprintf('Data: %d train, %d test, %d features\n', ...
    size(XTrain,1), size(XTest,1), size(XTrain,2));
