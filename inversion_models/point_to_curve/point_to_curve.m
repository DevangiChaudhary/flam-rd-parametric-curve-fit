clc; clear; close all;

%% --- Load and prepare data
data = csvread('xy_data.csv', 1, 0);
x_obs = data(:,1);
y_obs = data(:,2);

%% --- Parameter bounds
theta_min = 0; theta_max = 50;
M_min = -0.05; M_max = 0.05;
X_min = 0; X_max = 100;

lb = [theta_min, M_min, X_min];
ub = [theta_max, M_max, X_max];

%% --- Improved forward model
function [x_model, y_model] = forward_model(params, t)
    theta = params(1);
    M = params(2);
    X = params(3);
    x_model = t*cosd(theta) - exp(M*abs(t)).*sin(0.3*t).*sind(theta) + X;
    y_model = 42 + t*sind(theta) + exp(M*abs(t)).*sin(0.3*t).*cosd(theta);
end

%% --- Improved misfit function using point-to-curve distance
function dist = point_to_curve_misfit(params, x_obs, y_obs)
    % Sample the curve densely
    t_dense = linspace(6, 60, 5000);
    [x_curve, y_curve] = forward_model(params, t_dense);
    
    % For each observed point, find minimum distance to curve
    total_dist = 0;
    for i = 1:length(x_obs)
        distances = sqrt((x_curve - x_obs(i)).^2 + (y_curve - y_obs(i)).^2);
        total_dist = total_dist + min(distances);
    end
    dist = total_dist;
end

%% --- Alternative: Parameterize t for each point
function dist = parameter_misfit(params, x_obs, y_obs)
    % For each point, find best t that minimizes distance
    theta = params(1); M = params(2); X = params(3);
    
    total_dist = 0;
    for i = 1:length(x_obs)
        % Optimize t for this point
        t_opt = fminbnd(@(t) point_distance(t, theta, M, X, x_obs(i), y_obs(i)), 6, 60);
        [x_pred, y_pred] = forward_model(params, t_opt);
        total_dist = total_dist + abs(x_obs(i) - x_pred) + abs(y_obs(i) - y_pred);
    end
    dist = total_dist;
end

function d = point_distance(t, theta, M, X, x_target, y_target)
    [x_pred, y_pred] = forward_model([theta, M, X], t);
    d = abs(x_target - x_pred) + abs(y_target - y_pred);
end

%% --- Choose misfit function and optimize
misfit_func = @(params) point_to_curve_misfit(params, x_obs, y_obs);

nvars = 3;
options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', 500, ...
                      'MaxGenerations', 100, 'FunctionTolerance', 1e-6);

[x_opt, fval] = ga(misfit_func, nvars, [], [], [], [], lb, ub, [], options);

%% --- Refine with local search
options_fmin = optimset('Display', 'iter', 'TolFun', 1e-8);
[x_opt_refined, fval_refined] = fmincon(misfit_func, x_opt, [], [], [], [], lb, ub, [], options_fmin);

fprintf('Final parameters:\n');
fprintf('theta = %.6f deg, M = %.8f, X = %.6f\n', x_opt_refined);
fprintf('Total L1 misfit = %.6f\n', fval_refined);

%% --- Generate final curve and validate
t_fine = linspace(6, 60, 1000);
[x_final, y_final] = forward_model(x_opt_refined, t_fine);

figure;
scatter(x_obs, y_obs, 10, 'b', 'filled'); hold on;
plot(x_final, y_final, 'r-', 'LineWidth', 2);
xlabel('x'); ylabel('y');
title('Final Fit: Observed vs Predicted');
legend('Observed', 'Predicted', 'Location', 'best');
grid on;

% Calculate final L1 distance properly
final_L1 = point_to_curve_misfit(x_opt_refined, x_obs, y_obs);
fprintf('Validated L1 distance: %.6f\n', final_L1);