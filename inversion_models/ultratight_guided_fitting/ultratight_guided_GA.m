clc; clear; close all;

%% --- Load data and ULTRA-TIGHT guidance
data = csvread('xy_data.csv', 1, 0);
x_obs = data(:,1);
y_obs = data(:,2);

% Try ULTRA-TIGHT guidance first, fall back to regular guidance
if exist('ga_guidance_ultra_tight.mat', 'file')
    load('ga_guidance_ultra_tight.mat', 'guidance');
    fprintf('=== ULTRA-TIGHT SMART GA ===\n');
    fprintf('Using ULTRA-TIGHT guided ranges from analysis:\n');
elseif exist('ga_guidance.mat', 'file')
    load('ga_guidance.mat', 'guidance');
    fprintf('=== STANDARD SMART GA ===\n');
    fprintf('Using standard guided ranges from analysis:\n');
else
    fprintf('No guidance found. Running with full ranges.\n');
    guidance.theta_range = [0, 50];
    guidance.M_range = [-0.05, 0.05];
    guidance.X_range = [0, 100];
    guidance.best_start = [25, 0, 50];
    guidance.focus_parameter = 1;
end

fprintf('Theta: [%.1f, %.1f] deg (range: %.1f)\n', guidance.theta_range, diff(guidance.theta_range));
fprintf('M:     [%.4f, %.4f] (range: %.4f)\n', guidance.M_range, diff(guidance.M_range));
fprintf('X:     [%.1f, %.1f] (range: %.1f)\n', guidance.X_range, diff(guidance.X_range));

param_names = {'theta', 'M', 'X'};
fprintf('Focus parameter: %s\n', param_names{guidance.focus_parameter});
fprintf('Best starting point: theta=%.2f, M=%.4f, X=%.2f\n', guidance.best_start);

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
    t_dense = linspace(6, 60, 2000);  % Increased sampling for ultra-tight
    [x_curve, y_curve] = forward_model(params, t_dense);
    
    total_dist = 0;
    for i = 1:length(x_obs)
        distances = sqrt((x_curve - x_obs(i)).^2 + (y_curve - y_obs(i)).^2);
        total_dist = total_dist + min(distances);
    end
    dist = total_dist;
end

%% --- ULTRA-TIGHT Smart initial population
function initial_pop = create_ultra_tight_initial_pop(lb, ub, guidance, pop_size)
    initial_pop = zeros(pop_size, 3);
    
    % 80% in promising region, 20% random exploration (MORE FOCUSED)
    n_promising = round(0.8 * pop_size);
    
    for i = 1:pop_size
        if i <= n_promising
            % Sample from ULTRA-TIGHT promising region with Gaussian distribution
            % More samples near the center of the range
            theta_sample = guidance.theta_range(1) + randn() * 0.3 * diff(guidance.theta_range) + diff(guidance.theta_range)/2;
            M_sample = guidance.M_range(1) + randn() * 0.3 * diff(guidance.M_range) + diff(guidance.M_range)/2;
            X_sample = guidance.X_range(1) + randn() * 0.3 * diff(guidance.X_range) + diff(guidance.X_range)/2;
            
            % Ensure within bounds
            theta_sample = max(lb(1), min(ub(1), theta_sample));
            M_sample = max(lb(2), min(ub(2), M_sample));
            X_sample = max(lb(3), min(ub(3), X_sample));
            
            initial_pop(i, :) = [theta_sample, M_sample, X_sample];
        else
            % Random exploration in ultra-tight range
            initial_pop(i, :) = lb + rand(1,3) .* (ub - lb);
        end
    end
    
    % Ensure best starting point is included and multiple copies for diversity
    initial_pop(1, :) = guidance.best_start;
    initial_pop(2, :) = guidance.best_start + [0.1, 0.001, 0.1];  % Slight variation
    initial_pop(3, :) = guidance.best_start - [0.1, 0.001, 0.1];  % Slight variation
end

%% --- ULTRA-TIGHT Smart GA with guidance
function x_opt = ultra_tight_ga_with_guidance(guidance, x_obs, y_obs)
    lb = [guidance.theta_range(1), guidance.M_range(1), guidance.X_range(1)];
    ub = [guidance.theta_range(2), guidance.M_range(2), guidance.X_range(2)];
    
    fprintf('Optimizing in ULTRA-TIGHT parameter space...\n');
    
    % Calculate volume reduction
    full_ranges = [50, 0.1, 100];
    guided_ranges = ub - lb;
    volume_reduction = (1 - prod(guided_ranges) / prod(full_ranges)) * 100;
    fprintf('Search volume reduced by %.1f%%\n', volume_reduction);
    
    % SMALLER population for ultra-tight ranges
    pop_size = 120;  % Reduced from 200
    
    % Create ULTRA-TIGHT initial population FIRST
    initial_population = create_ultra_tight_initial_pop(lb, ub, guidance, pop_size);
    
    % Enhanced GA options for ULTRA-TIGHT ranges
    switch guidance.focus_parameter
        case 1  % Theta focused - more exploration in theta
            options = optimoptions('ga', ...
                'Display', 'iter', ...
                'PopulationSize', pop_size, ...
                'MaxGenerations', 50, ...  % Fewer generations needed
                'FunctionTolerance', 1e-7, ...  % Tighter tolerance
                'CrossoverFraction', 0.85, ...  % More crossover
                'InitialPopulationMatrix', initial_population);
            
        case 2  % M focused - very tight tolerance for M
            options = optimoptions('ga', ...
                'Display', 'iter', ...
                'PopulationSize', pop_size, ...
                'MaxGenerations', 45, ...  % Even fewer generations
                'FunctionTolerance', 1e-8, ...  % Very tight tolerance
                'CrossoverFraction', 0.75, ...
                'InitialPopulationMatrix', initial_population);
            
        case 3  % X focused
            options = optimoptions('ga', ...
                'Display', 'iter', ...
                'PopulationSize', pop_size, ...
                'MaxGenerations', 55, ...
                'FunctionTolerance', 1e-7, ...
                'CrossoverFraction', 0.8, ...
                'InitialPopulationMatrix', initial_population);
    end
    
    fprintf('Running ULTRA-TIGHT GA with %d population, %d generations\n', ...
            pop_size, options.MaxGenerations);
    
    nvars = 3;
    x_opt = ga(@(p) point_to_curve_misfit(p, x_obs, y_obs), nvars, ...
               [], [], [], [], lb, ub, [], options);
end

%% --- Enhanced final refinement for ULTRA-TIGHT
function [x_refined, fval_refined] = ultra_tight_refinement(x_start, x_obs, y_obs)
    fprintf('Running ULTRA-TIGHT refinement...\n');
    
    lb_full = [0, -0.05, 0];
    ub_full = [50, 0.05, 100];
    
    % Two-stage refinement for better convergence
    % Stage 1: Medium precision
    options_medium = optimoptions('fmincon', ...
        'Display', 'off', ...  % Quiet for first stage
        'Algorithm', 'interior-point', ...
        'FunctionTolerance', 1e-7, ...
        'StepTolerance', 1e-7, ...
        'MaxFunctionEvaluations', 150);
    
    [x_medium, fval_medium] = fmincon(@(p) point_to_curve_misfit(p, x_obs, y_obs), ...
                                     x_start, [], [], [], [], lb_full, ub_full, [], options_medium);
    
    fprintf('Medium refinement: L1 = %.6f\n', fval_medium);
    
    % Stage 2: High precision
    options_high = optimoptions('fmincon', ...
        'Display', 'iter', ...
        'Algorithm', 'sqp', ...
        'FunctionTolerance', 1e-10, ...  % ULTRA-TIGHT tolerance
        'StepTolerance', 1e-10, ...
        'MaxFunctionEvaluations', 200);
    
    [x_refined, fval_refined] = fmincon(@(p) point_to_curve_misfit(p, x_obs, y_obs), ...
                                       x_medium, [], [], [], [], lb_full, ub_full, [], options_high);
    
    fprintf('High refinement: L1 = %.6f\n', fval_refined);
end

%% --- Run ULTRA-TIGHT smart GA
x_opt = ultra_tight_ga_with_guidance(guidance, x_obs, y_obs);
fprintf('GA completed. Best found: theta=%.4f, M=%.6f, X=%.4f\n', x_opt);

%% --- ULTRA-TIGHT Final refinement
[x_opt_refined, fval_refined] = ultra_tight_refinement(x_opt, x_obs, y_obs);

%% --- Enhanced results analysis
fprintf('\n=== ULTRA-TIGHT FINAL RESULTS ===\n');
fprintf('theta = %.8f deg\n', x_opt_refined(1));
fprintf('M     = %.10f\n', x_opt_refined(2));
fprintf('X     = %.8f\n', x_opt_refined(3));
fprintf('Final L1 distance: %.8f\n', fval_refined);

% Enhanced validation with denser sampling
final_L1 = point_to_curve_misfit(x_opt_refined, x_obs, y_obs);
fprintf('Validated L1 (2000 t-points): %.8f\n', final_L1);

% Improvement analysis
initial_L1 = point_to_curve_misfit(guidance.best_start, x_obs, y_obs);
improvement = ((initial_L1 - final_L1) / initial_L1) * 100;
fprintf('Improvement from starting point: %.2f%%\n', improvement);

%% --- Enhanced plot with error visualization
t_fine = linspace(6, 60, 2000);
[x_final, y_final] = forward_model(x_opt_refined, t_fine);

figure('Position', [100, 100, 1400, 500]);

% Plot 1: Curve fit
subplot(1,2,1);
scatter(x_obs, y_obs, 15, 'b', 'filled'); hold on;
plot(x_final, y_final, 'r-', 'LineWidth', 2);
xlabel('x'); ylabel('y');
title(sprintf('ULTRA-TIGHT GA Result (L1 = %.6f)', final_L1));
legend('Observed', 'Predicted', 'Location', 'best');
grid on;
axis equal;

% Parameter bounds visualization
fprintf('\n=== PARAMETER BOUNDS USED ===\n');
fprintf('Theta: %.1f to %.1f (optimal: %.3f)\n', guidance.theta_range, x_opt_refined(1));
fprintf('M:     %.4f to %.4f (optimal: %.6f)\n', guidance.M_range, x_opt_refined(2));
fprintf('X:     %.1f to %.1f (optimal: %.3f)\n', guidance.X_range, x_opt_refined(3));

% Check if optimal is near bounds (might indicate need for wider ranges)
theta_margin = min(x_opt_refined(1) - guidance.theta_range(1), guidance.theta_range(2) - x_opt_refined(1));
M_margin = min(x_opt_refined(2) - guidance.M_range(1), guidance.M_range(2) - x_opt_refined(2));
X_margin = min(x_opt_refined(3) - guidance.X_range(1), guidance.X_range(2) - x_opt_refined(3));

fprintf('\nParameter margin to bounds:\n');
fprintf('Theta margin: %.2f deg\n', theta_margin);
fprintf('M margin:     %.6f\n', M_margin);
fprintf('X margin:     %.2f units\n', X_margin);

if theta_margin < 1 || M_margin < 0.002 || X_margin < 5
    fprintf('-> WARNING: Optimal solution is near bounds. Consider wider ranges.\n');
else
    fprintf('-> GOOD: Optimal solution has sufficient margin from bounds.\n');
end