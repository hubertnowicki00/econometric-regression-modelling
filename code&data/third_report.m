%% REPORT TASK 3.1
% Verify the autoregressive model:
% yt = Beta[1]*y[t-1]+Beta[2]*y[t-2]+Beta[3]*y[t-7]+epsilon
% Use alpha = 1% significance level
% Check the autocorrelation of the residuals for 10 first lags
% Check the collinearity
% Interpret the results

clear variables
clc

t = load('263807_third_report\report_data3_1.mat');
t_clear = t.y(:);

Y = t_clear(8:end);
X(:,1) = t_clear(7:end-1); % Beta[1]*Y[t-1]
X(:,2) = t_clear(6:end-2); % Beta[2]*Y[t-2]
X(:,3) = t_clear(1:end-7); % Beta[3]*Y[t-7]


N = length(Y);
K = size(X,2);
beta = OLS(X,Y);
e = Y - X*beta; % model errors / residuals

% 10 lags autocorrelation | q-test for autocorrelation
maxlag = 10;
alpha = 0.01;
a = autocorr(e, 'NumLags', maxlag);
a = a(2:end);

% H0 : autocorrelation coeffs until lag maxlag are 0
Q = N*(N+2)*sum(a.^2 ./ (N-(1:maxlag))');
p_val_q = 1 - chi2cdf(Q, maxlag);

if p_val_q < alpha
    disp('Autocorrelation present in data')
else
    disp('No autocorrelation present in data')
end

% LM | Auxillary Regression
y_aux = e(maxlag+1:end);
E = zeros(length(y_aux), maxlag);
for i = 1:maxlag

    E(:,i) = e(maxlag+1-i:end-i);

end

X_aux = [X(maxlag+1:end,:), E]; % matrix for the auxiliary regression
betas_aux = OLS(X_aux, y_aux);
e_aux = y_aux - X_aux*betas_aux;
r2_lm = 1 - var(e_aux)/var(y_aux);

% test statistics
LM = r2_lm*length(e_aux);
p_val_lm = 1 - chi2cdf(LM, maxlag);

if p_val_lm < alpha
    disp('Autocorrelation present in data')
else
    disp('No autocorrelation present in data')
end

% collinearity
X_t = X - mean(X);
VIF = zeros(1, K);

for k = 1:K

    idx = 1:K;
    idx(k) = [];
    y_aux_vif = X_t(:,k);
    x_aux_vif = X_t(:,idx);
    
    betas_vif = OLS(x_aux_vif, y_aux_vif);
    e_aux_vif = y_aux_vif - x_aux_vif*betas_vif;
    
    r2_vif = 1 - var(e_aux_vif)/var(y_aux_vif);
    VIF(k) = 1/(1-r2_vif);

    fprintf('VIF for Y[t-%d]: %.4f\n', k, VIF(k));

end

% eigenvalues of the matrix
eigs = eig(X_t'*X_t); % 0.0329; 0.2435; 1.3094
CI = sqrt(max(eigs)./eigs); % 6.3084; 2.3189; 1.0000
ConditionNumber = max(CI);

% VIF interpretation
if any(VIF > 10)
    disp('VIF number is greater than 10, thus multicollinearity exists in data')
else
    disp('VIF number is lower than 10, thus multicollinearity does not exist / is weak')
end

% CI interpretation
if ConditionNumber > 30
    disp('CI bigger greater than 30, thus multicollinearity exists in data')
elseif ConditionNumber > 15
    disp('CI between 15 and 30, thus multicollinearity is somewhat strong')
else
    disp('CI lower than 15, thus multicollinearity does not exist / is weak')
end




%% REPORT TASK 3.2
% Verify the model:
% Price[t] =
% Beta[1]*Price[t-1]+Beta[2]*Consumption[t]+Beta[3]*Consumption[t-1]+e
% Use alpha = 1% significance level. If there are more than one method of
% verification use all of them
% Check the homoscedasticity
% Check the stability of the parameters
% Interpret the results

clear variables
clc

t = load('263807_third_report\report_data3_2.mat'); 
Price = t.Price(:); 
Consumption = t.Consumption(:);

Y = Price(2:end);
N = length(Y);
X = zeros(N, 3);
X(:,1) = Price(1:end-1);       % beta[1] * Price[t-1]
X(:,2) = Consumption(2:end);   % beta[2] * Consumption
X(:,3) = Consumption(1:end-1); % beta[3] * Consumption[t-1]

OriginalK = size(X,2); % we have 3 original variables

beta = OLS(X,Y);
e = Y - X*beta; % model errors / residuals

% we will use both White and Breusch-Pagan LM tests
% (both of them are also viable method of verification)

% White-test
x_white = X;
for i = 1:size(X,2)

    for j = i:size(X,2)

        x_white = [x_white, X(:,i).*X(:,j)]; 

    end

end
x_white(:, end+1) = 1;

betas_2 = OLS(x_white, e.^2);
e2 = e.^2 - x_white*betas_2; % auxiliary model errors
r2 = 1 - var(e2)/var(e.^2);
stat_w = r2*N;
m = OriginalK + OriginalK*(OriginalK+1)/2; 
pval_white = 1 - chi2cdf(stat_w, m);

% Breusch-Pagan test
g = e.^2 / (e'*e/N) - 1;
Z = X; 
stat_bp = 0.5 * g' * Z * inv(Z'*Z) * Z' * g; 
pval_bp = 1 - chi2cdf(stat_bp, OriginalK);

% parameters stability - cusumtest
cusumtest(X, Y);

% interpretation
alpha = 0.01;

% White-test
if pval_white > alpha
    disp(['WHITE: We cannot reject the null hypothesis, thus proving the residuals ' ...
        'are homoscedastic']);
else
    disp(['WHITE: We do not have grounds to not reject the null hypothesis, thus ' ...
        'proving the residuals are heteroscedastic        ']);
end

% Breusch-Pagan test
if pval_bp > alpha
    disp(['BREUSCH-PAGAN: We cannot reject the null hypothesis, thus proving the residuals ' ...
        'are homoscedastic']);
else
    disp(['BREUSCH-PAGAN: We do not have grounds to not reject the null hypothesis, thus ' ...
        'proving the residuals are heteroscedastic        ']);
end


%% REPORT TASK 3.3 - 5 points variant


clear variables
clc

t = readtable('263807_third_report/report_data3_3.csv');
hubert_sex = 'male';

% filtering for my parameters - bday in October (10th month of the year)
t = t(strcmp(t.sex, hubert_sex) & month(t.birthday) == 10, :);
Y = t.wage(:);
N = length(Y);

X = zeros(N, 5);
X(:,1) = t.Experience(:);
X(:,2) = t.Skills(:);
X(:,3) = t.Years_in_company(:);
X(:,4) = t.supervisors_assessment(:);
X(:,5) = t.motivation(:);

OriginalK_2 = size(X,2);
variable_names = {'Experience', 'Skills', 'Years in Company', 'Supervisor Assessment', 'Motivation'};

beta_init = OLS(X, Y);
e_init = Y - X*beta_init; % model errors / residuals

% autocorrelation | LM test | auxillary
p_lags = 20;
E_lag = zeros(N - p_lags, p_lags);

for i = 1:p_lags
    E_lag(:, i) = e_init(p_lags - i + 1 : end - i);
end

X_lm = [X(p_lags+1:end, :), E_lag];
e_curr = e_init(p_lags+1:end);

beta_lm = (X_lm'*X_lm)^-1 * X_lm' * e_curr;
e_lm_aux = e_curr - X_lm*beta_lm;
R2_lm = 1 - (e_lm_aux'*e_lm_aux) / (e_curr'*e_curr);

stat_lm_autocorr = (N - p_lags) * R2_lm;
pval_autocorr_lm = 1 - chi2cdf(stat_lm_autocorr, p_lags);

% homoscedasticity using White test
x_white = X;

for i = 1:OriginalK_2

    for j = i:OriginalK_2

        x_white = [x_white, X(:,i).*X(:,j)];

    end

end

x_white(:, end+1) = 1; 
betas_white = OLS(x_white, e_init.^2);

e2_white = e_init.^2 - x_white*betas_white;
r2_white = 1 - var(e2_white)/var(e_init.^2);

stat_w = r2_white * N;
m_white = OriginalK_2 + OriginalK_2*(OriginalK_2+1)/2; 
pval_white = 1 - chi2cdf(stat_w, m_white);

if pval_white > 0.01
    disp(['WHITE: We cannot reject the null hypothesis, thus proving the residuals' ...
        'are homoscedastic']);
else
    disp(['WHITE: We do not have grounds to not reject the null hypothesis, thus' ...
        'proving the residuals are heteroscedastic        ']);
end

% VIF collinearity
VIFs = zeros(1, OriginalK_2);

for i = 1:OriginalK_2

    y_vif = X(:, i);
    X_vif = X;
    X_vif(:, i) = [];

    beta_vif = (X_vif'*X_vif)^-1 * X_vif' * y_vif;
    e_vif = y_vif - X_vif*beta_vif;

    R2_vif = 1 - (e_vif'*e_vif) / (y_vif'*y_vif);
    VIFs(i) = 1 / (1 - R2_vif);

end

% VIF interpretation
if any(VIFs > 10)
    disp('VIF number is greater than 10, thus multicollinearity exists in data')

    corr_matrix = corr(X);
    corr_matrix(logical(eye(size(corr_matrix)))) = 0;

    [~, max_idx] = max(abs(corr_matrix(:)));
    [row, col] = ind2sub(size(corr_matrix), max_idx);

    fprintf('Correlated variables -> %s and %s\n', variable_names{row}, variable_names{col});
else
    disp('VIF number is lower than 10, thus multicollinearity does not exist / is weak')
end

% forward stepwise regression
included_vars = [];
available_vars = 1:OriginalK_2;

while true

    best_pval = 1;
    best_var = 0;
    
    for i = available_vars

        cols_to_test = [included_vars, i];
        X_test = X(:, cols_to_test);
        
        [beta_test, ~, se_test] = OLS(X_test, Y);
        
        % t-test with pval
        t_stat = beta_test(end) / se_test(end);
        df = N - length(cols_to_test);
        pval = 2 * (1 - tcdf(abs(t_stat), df));
        
        if pval < best_pval

            best_pval = pval;
            best_var = i;

        end
    end
    
    if best_pval < 0.05 % threshold was specified to be 5%

        included_vars = [included_vars, best_var];
        available_vars(available_vars == best_var) = [];
        fprintf('Added variable %d (%s) with p-value: %.4f\n', best_var, variable_names{best_var}, best_pval);
    
    else
        break;
    end
end

% verifying newly obtained model
% autocorrelation & homoscedascity & collinearity
X_new = X(:, included_vars);
K_new = size(X_new, 2);

beta_new = OLS(X_new, Y);
e_new = Y - X_new*beta_new;

% autocorrelation Q-test
Q_stat = 0;
for k = 1:p_lags

    r_k = corr(e_new(k+1:end), e_new(1:end-k));
    Q_stat = Q_stat + (r_k^2) / (N - k);

end

Q_stat = N * (N + 2) * Q_stat;
pval_autocorr_q = 1 - chi2cdf(Q_stat, p_lags);

% homoscedascity with Breusch-Pagan LM test
g = e_new.^2 / ((e_new'*e_new)/N) - 1;
Z = X_new;

stat_bp = 0.5 * g' * Z * inv(Z'*Z) * Z' * g;
pval_bp = 1 - chi2cdf(stat_bp, K_new);

% collinearity
% CI[i] = sqrt(lambda[max]/lambda[i])
X_new_t = X_new - mean(X_new); 
eigs_new = eig(X_new_t' * X_new_t);
CI_new = sqrt(max(eigs_new)./eigs_new);
cond_num = max(CI_new);

% results interpretation of the old model
disp(' ');
disp('"OLD" MODEL VERIFICATION');
fprintf('Autocorrelation p-val -> %.4f\n', pval_autocorr_lm);
fprintf('Homoscedasticity p-val -> %.4f\n', pval_white);
disp('VIFs -> '); disp(VIFs);

% max significance level
max_alpha_init = min(pval_autocorr_lm, pval_white);
fprintf('Maximum alpha value for homoscedastic & autocorrelated residuals -> %.4f\n', max_alpha_init); 
 
if any(VIFs > 10)
    disp('VIF number is greater than 10, thus multicollinearity exists in data')
else
    disp('VIF number is lower than 10, thus multicollinearity does not exist / is weak')
end

disp(' ');
disp('NEWLY OBTAINED MODEL VERIFICATION');
fprintf('Variables included -> '); disp(included_vars);
fprintf('Autocorrelation p-val -> %.4f\n', pval_autocorr_q);
fprintf('Homoscedasticity p-val -> %.4f\n', pval_bp);
fprintf('CI -> %.4f\n', cond_num);

% max significance level
max_alpha_new = min(pval_autocorr_q, pval_bp);
fprintf('Maximum alpha value for homoscedastic & autocorrelated residuals -> %.4f\n', max_alpha_new);

if cond_num > 30
    disp('CI bigger greater than 30, thus multicollinearity exists in data')
elseif cond_num > 15
    disp('CI between 15 and 30, thus multicollinearity is somewhat strong')
else
    disp('CI lower than 15, thus multicollinearity does not exist / is weak')
end