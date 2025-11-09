clc; clear; close all;

%% --- Load data and guidance
data = csvread('xy_data.csv', 1, 0);
x_obs = data(:,1);
y_obs = data(:,2);

if exist('ga_guidance.mat', 'file')
    load('ga_guidance.mat', 'guidance');
    fprintf('=== SMART GA USING JACOBIAN INSIGHTS ===\n');
    fprintf('Using guided ranges from analysis:\n');
    fprintf('Theta: [%.1f, %.1f] deg\n', guidance.theta_range);
    fprintf('M:     [%.4f, %.4f]\n', guidance.M_range);
    fprintf('X:     [%.1f, %.1f]\n', guidance.X_range);
    
    % FIXED: Proper cell array indexing
    param_names = {'theta', 'M', 'X'};
    fprintf('Focus parameter: %s\n', param_names{guidance.focus_parameter});
else
    fprintf('No guidance found. Running with full ranges.\n');
    guidance.theta_range = [0, 50];
    guidance.M_range = [-0.05, 0.05];
    guidance.X_range = [0, 100];
    guidance.best_start = [25, 0, 50];
    guidance.focus_parameter = 1; % Add this missing field
end

%% --- Forward model
function [x_model, y_model] = forward_model(params, t)
    theta = params(1);
    M = params(2);
    X = params(3);
    x_model = t*cosd(theta) - exp(M*abs(t)).*sin(0.3*t).*sind(theta) + X;
    y_model = 42 + t*sind(theta) + exp(M*abs(t)).*sin(0.3*t).*cosd(theta);
end

%% --- Misfit function
function dist = point_to_curve_misfit(params, x_obs, y_obs)
    t_dense = linspace(6, 60, 1500);
    [x_curve, y_curve] = forward_model(params, t_dense);
    
    total_dist = 0;
    for i = 1:length(x_obs)
        distances = sqrt((x_curve - x_obs(i)).^2 + (y_curve - y_obs(i)).^2);
        total_dist = total_dist + min(distances);
    end
    dist = total_dist;
end

%% --- Smart initial population
function initial_pop = create_smart_initial_pop(lb, ub, guidance, pop_size)
    initial_pop = zeros(pop_size, 3);
    
    % 60% in promising region, 40% random exploration
    n_promising = round(0.6 * pop_size);
    
    for i = 1:pop_size
        if i <= n_promising
            % Sample from promising region
            initial_pop(i, :) = [...
                guidance.theta_range(1) + rand() * diff(guidance.theta_range), ...
                guidance.M_range(1) + rand() * diff(guidance.M_range), ...
                guidance.X_range(1) + rand() * diff(guidance.X_range)];
        else
            % Random exploration in full range
            initial_pop(i, :) = lb + rand(1,3) .* (ub - lb);
        end
    end
    
    % Ensure best starting point is included
    initial_pop(1, :) = guidance.best_start;
end

%% --- Smart GA with guidance
function x_opt = smart_ga_with_guidance(guidance, x_obs, y_obs)
    lb = [guidance.theta_range(1), guidance.M_range(1), guidance.X_range(1)];
    ub = [guidance.theta_range(2), guidance.M_range(2), guidance.X_range(2)];
    
    fprintf('Optimizing in focused parameter space...\n');
    
    % Calculate volume reduction properly
    full_ranges = [50, 0.1, 100];
    guided_ranges = ub - lb;
    volume_reduction = (1 - prod(guided_ranges) / prod(full_ranges)) * 100;
    fprintf('Search volume reduced by %.1f%%\n', volume_reduction);
    
    % Custom GA options based on sensitivity
    pop_size = 200;
    
    switch guidance.focus_parameter
        case 1  % Theta focused
            options = optimoptions('ga', ...
                'Display', 'iter', ...
                'PopulationSize', pop_size, ...
                'MaxGenerations', 80, ...
                'FunctionTolerance', 1e-6, ...
                'CrossoverFraction', 0.8);
            
        case 2  % M focused  
            options = optimoptions('ga', ...
                'Display', 'iter', ...
                'PopulationSize', pop_size, ...
                'MaxGenerations', 60, ...
                'FunctionTolerance', 1e-7, ...
                'CrossoverFraction', 0.7);
            
        case 3  % X focused
            options = optimoptions('ga', ...
                'Display', 'iter', ...
                'PopulationSize', pop_size, ...
                'MaxGenerations', 70, ...
                'FunctionTolerance', 1e-6);
    end
    
    % Create smart initial population
    initial_population = create_smart_initial_pop(lb, ub, guidance, pop_size);
    options.InitialPopulationMatrix = initial_population;
    
    nvars = 3;
    x_opt = ga(@(p) point_to_curve_misfit(p, x_obs, y_obs), nvars, ...
               [], [], [], [], lb, ub, [], options);
end

%% --- Run smart GA
x_opt = smart_ga_with_guidance(guidance, x_obs, y_obs);

%% --- Final refinement
fprintf('\n=== FINAL REFINEMENT ===\n');
options_fmin = optimoptions('fmincon', 'Display', 'iter', 'FunctionTolerance', 1e-8);
[x_opt_refined, fval_refined] = fmincon(@(p) point_to_curve_misfit(p, x_obs, y_obs), ...
                                       x_opt, [], [], [], [], ...
                                       [0, -0.05, 0], [50, 0.05, 100], [], options_fmin);

%% --- Results
fprintf('\n=== FINAL RESULTS ===\n');
fprintf('theta = %.6f deg, M = %.8f, X = %.6f\n', x_opt_refined);
fprintf('Final L1 distance: %.6f\n', fval_refined);

% Validation with dense sampling
final_L1 = point_to_curve_misfit(x_opt_refined, x_obs, y_obs);
fprintf('Validated L1: %.6f\n', final_L1);

%% --- Plot final result
t_fine = linspace(6, 60, 1000);
[x_final, y_final] = forward_model(x_opt_refined, t_fine);

figure;
scatter(x_obs, y_obs, 10, 'b', 'filled'); hold on;
plot(x_final, y_final, 'r-', 'LineWidth', 2);
xlabel('x'); ylabel('y');
title(sprintf('Smart GA Result (L1 = %.4f)', final_L1));
legend('Observed', 'Predicted', 'Location', 'best');
grid on;