clc; clear; close all;

%% --- Load observed data
data = csvread('C:\Users\Devangi Chaudhary\Desktop\Flam R&D\xy_data.csv', 1, 0);
x_obs = data(:,1);
y_obs = data(:,2);
t_all = linspace(6, 60, length(x_obs));  % full t vector

%% --- Parameter bounds
theta_min = 0; theta_max = 50;   % degrees
M_min     = -0.05; M_max     = 0.05;
X_min     = 0; X_max = 100;

lb = [theta_min, M_min, X_min];
ub = [theta_max, M_max, X_max];

%% --- Forward model returning single concatenated vector
function XY = forward_model_vec(params, t)
    theta = params(1);
    M     = params(2);
    X     = params(3);
    x_model = t*cosd(theta) - exp(M*abs(t)).*sin(0.3*t).*sind(theta) + X;
    y_model = 42 + t*sind(theta) + exp(M*abs(t)).*sin(0.3*t).*cosd(theta);
    XY = [x_model(:); y_model(:)];  % concatenate as a column vector
end

%% --- Misfit function (scalar) for GA
misfit = @(params) sum(abs([x_obs(:); y_obs(:)] - forward_model_vec(params, t_all)));

%% --- Genetic Algorithm optimization
nvars = 3;  % [theta, M, X]
options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', 1000);

[x_opt, fval] = ga(misfit, nvars, [], [], [], [], lb, ub, [], options);

theta_opt = x_opt(1);
M_opt     = x_opt(2);
X_opt     = x_opt(3);

fprintf('Optimal parameters found:\n');
fprintf('theta = %.4f deg, M = %.5f, X = %.4f\n', theta_opt, M_opt, X_opt);
fprintf('Total L1 misfit = %.4f\n', fval);

%% --- Compute predicted curve using optimal parameters
XY_pred = forward_model_vec(x_opt, t_all);
x_pred = XY_pred(1:length(t_all));
y_pred = XY_pred(length(t_all)+1:end);

%% --- Plot observed vs predicted curve
figure;
scatter(x_obs, y_obs, 10, 'b', 'filled'); hold on;
plot(x_pred, y_pred, 'r', 'LineWidth', 2);
xlabel('x'); ylabel('y');
title('Observed points (blue) vs Predicted curve (red)');
grid on; legend('Observed', 'Predicted');


%% --- Plot x vs t and y vs t using predicted parameters
figure;

% x vs t

scatter(t_all, x_pred, 10, 'b', 'filled');  % scatter plot
xlabel('t'); ylabel('x (predicted)');
title('Predicted x vs t');
grid on;

% y vs t
figure;
scatter(t_all, y_pred, 10, 'r', 'filled');  % scatter plot
xlabel('t'); ylabel('y (predicted)');
title('Predicted y vs t');
grid on;

%%
%% --- Compute true L1 distance after GA (as per problem statement)
L1_true = sum(abs(x_pred - x_obs)) + sum(abs(y_pred - y_obs));
L1_euclid = sum( sqrt( (x_pred - x_obs).^2 + (y_pred - y_obs).^2 ) );

fprintf('\n---------------------------------------------\n');
fprintf('GA-computed misfit (objective function) = %.4f\n', fval);
fprintf('True L1 curve distance (expected vs predicted) = %.4f\n', L1_true);
fprintf('True L1 ecurve distance (expected vs predicted) = %.4f\n', L1_euclid);
fprintf('---------------------------------------------\n\n');