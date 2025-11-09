clc; clear; close all;

%% --- Load data
data = csvread('xy_data.csv', 1, 0);
x_obs = data(:,1);
y_obs = data(:,2);

%% --- Forward model with Jacobian
function [x_model, y_model, dx_dparams, dy_dparams] = forward_model_with_jacobian(params, t)
    theta = params(1);
    M = params(2);
    X = params(3);
    
    % Precompute common terms
    cos_t = cosd(theta);
    sin_t = sind(theta);
    abs_t = abs(t);
    exp_term = exp(M * abs_t);
    sin_03t = sin(0.3 * t);
    
    % Forward model
    x_model = t*cos_t - exp_term.*sin_03t.*sin_t + X;
    y_model = 42 + t*sin_t + exp_term.*sin_03t.*cos_t;
    
    % Jacobian components
    if nargout > 2
        n_t = length(t);
        
        % Derivatives w.r.t. theta
        dcos_dtheta = -sind(theta);
        dsin_dtheta = cosd(theta);
        
        dx_dtheta = t * dcos_dtheta - exp_term .* sin_03t .* dsin_dtheta;
        dy_dtheta = t * dsin_dtheta + exp_term .* sin_03t .* dcos_dtheta;
        
        % Derivatives w.r.t. M
        dexp_dM = abs_t .* exp_term;
        dx_dM = -dexp_dM .* sin_03t .* sin_t;
        dy_dM = dexp_dM .* sin_03t .* cos_t;
        
        % Derivatives w.r.t. X
        dx_dX = ones(n_t, 1);
        dy_dX = zeros(n_t, 1);
        
        dx_dparams = [dx_dtheta(:), dx_dM(:), dx_dX(:)];
        dy_dparams = [dy_dtheta(:), dy_dM(:), dy_dX(:)];
    end
end

%% --- Misfit function
function misfit = calculate_misfit(params, x_obs, y_obs)
    t_dense = linspace(6, 60, 1500);
    [x_curve, y_curve] = forward_model_with_jacobian(params, t_dense);
    
    total_dist = 0;
    for i = 1:length(x_obs)
        distances = sqrt((x_curve - x_obs(i)).^2 + (y_curve - y_obs(i)).^2);
        total_dist = total_dist + min(distances);
    end
    misfit = total_dist;
end

%% --- Jacobian at point
function [f, J] = jacobian_at_point(params, x_obs, y_obs)
    t_guess = linspace(6, 60, length(x_obs));
    [x_pred, y_pred, dx_dp, dy_dp] = forward_model_with_jacobian(params, t_guess);
    
    f = [x_obs - x_pred; y_obs - y_pred];
    J = [-dx_dp; -dy_dp];
end

%% --- Latin Hypercube Sampling
function samples = get_latin_hypercube_samples(n_samples)
    lb = [0, -0.05, 0];
    ub = [50, 0.05, 100];
    
    samples = lhsdesign(n_samples, 3);
    for i = 1:3
        samples(:, i) = lb(i) + samples(:, i) * (ub(i) - lb(i));
    end
end

%% --- Visualization
function visualize_parameter_space(theta_vals, M_vals, X_vals, misfits, good_indices, x_obs, y_obs)
    figure('Position', [100, 100, 1200, 900]);
    
    % Plot 1: Theta vs M
    subplot(2,3,1);
    scatter(theta_vals, M_vals, 30, misfits, 'filled');
    hold on;
    scatter(theta_vals(good_indices), M_vals(good_indices), 50, 'r', 'linewidth', 1.5);
    colorbar;
    xlabel('Theta (deg)'); ylabel('M');
    title('Theta vs M (color = L1 misfit)');
    grid on;
    
    % Plot 2: Theta vs X
    subplot(2,3,2);
    scatter(theta_vals, X_vals, 30, misfits, 'filled');
    hold on;
    scatter(theta_vals(good_indices), X_vals(good_indices), 50, 'r', 'linewidth', 1.5);
    colorbar;
    xlabel('Theta (deg)'); ylabel('X');
    title('Theta vs X (color = L1 misfit)');
    grid on;
    
    % Plot 3: M vs X
    subplot(2,3,3);
    scatter(M_vals, X_vals, 30, misfits, 'filled');
    hold on;
    scatter(M_vals(good_indices), X_vals(good_indices), 50, 'r', 'linewidth', 1.5);
    colorbar;
    xlabel('M'); ylabel('X');
    title('M vs X (color = L1 misfit)');
    grid on;
    
    % Plot 4: Misfit distribution
    subplot(2,3,4);
    histogram(misfits, 30, 'FaceColor', 'blue', 'FaceAlpha', 0.7);
    hold on;
    yl = ylim;
    plot([prctile(misfits, 25), prctile(misfits, 25)], yl, 'r--', 'linewidth', 2);
    xlabel('L1 Misfit'); ylabel('Frequency');
    title('Misfit Distribution');
    legend('All points', '25th percentile');
    grid on;
    
    % Plot 5: Best solution preview
    subplot(2,3,5);
    [best_misfit, best_idx] = min(misfits);
    best_params = [theta_vals(best_idx), M_vals(best_idx), X_vals(best_idx)];
    
    t_fine = linspace(6, 60, 1000);
    [x_best, y_best] = forward_model_with_jacobian(best_params, t_fine);
    
    scatter(x_obs, y_obs, 10, 'b', 'filled'); hold on;
    plot(x_best, y_best, 'r-', 'LineWidth', 2);
    xlabel('x'); ylabel('y');
    title(sprintf('Best Found (L1=%.1f)', best_misfit));
    legend('Observed', 'Predicted');
    grid on;
    
    % Plot 6: Parameter correlations
    subplot(2,3,6);
    correlations = corr([theta_vals, M_vals, X_vals, misfits]);
    imagesc(correlations(1:3, 4));
    colorbar;
    set(gca, 'XTick', 1:3, 'XTickLabel', {'Theta', 'M', 'X'});
    ylabel('Correlation with Misfit');
    title('Parameter-Misfit Correlations');
    
    for i = 1:3
        text(i, 1, sprintf('%.3f', correlations(i,4)), ...
             'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
    
    sgtitle('Comprehensive Parameter Space Analysis');
end

%% --- Create GA guidance
function create_GA_guidance(theta_vals, M_vals, X_vals, theta_sens, M_sens, X_sens, sensitivity_map)
    fprintf('\n=== GA OPTIMIZATION GUIDANCE ===\n');
    
    % Calculate recommended ranges (mean ± 1.5*std for tighter focus)
    theta_range = [max(0, mean(theta_vals) - 1.5*std(theta_vals)), ...
                   min(50, mean(theta_vals) + 1.5*std(theta_vals))];
    M_range = [max(-0.05, mean(M_vals) - 1.5*std(M_vals)), ...
               min(0.05, mean(M_vals) + 1.5*std(M_vals))];
    X_range = [max(0, mean(X_vals) - 1.5*std(X_vals)), ...
               min(100, mean(X_vals) + 1.5*std(X_vals))];
    
    fprintf('Recommended GA search ranges:\n');
    fprintf('theta: [%.1f, %.1f] deg\n', theta_range);
    fprintf('M:     [%.4f, %.4f]\n', M_range);
    fprintf('X:     [%.1f, %.1f]\n', X_range);
    
    % Find most sensitive parameter
    sensitivities = [mean(theta_sens), mean(M_sens), mean(X_sens)];
    [~, focus_param] = max(sensitivities);
    param_names = {'theta', 'M', 'X'};
    
    fprintf('\nOptimization strategy:\n');
    fprintf('-> Focus mutations on %s (most sensitive parameter)\n', param_names{focus_param});
    
    % Find best starting point from analysis
    fields = fieldnames(sensitivity_map);
    best_misfit = inf;
    best_start = [];
    
    for i = 1:length(fields)
        data = sensitivity_map.(fields{i});
        if data.misfit < best_misfit
            best_misfit = data.misfit;
            best_start = data.params;
        end
    end
    
    fprintf('-> Best starting point: theta=%.2f, M=%.4f, X=%.2f (L1=%.1f)\n', ...
            best_start, best_misfit);
    
    % Save guidance
    guidance = struct();
    guidance.theta_range = theta_range;
    guidance.M_range = M_range;
    guidance.X_range = X_range;
    guidance.focus_parameter = focus_param;
    guidance.best_start = best_start;
    guidance.best_start_misfit = best_misfit;
    
    save('ga_guidance.mat', 'guidance');
    fprintf('\nGuidance saved to ga_guidance.mat\n');
end

%% --- Analyze results
function analyze_sensitivity_results(sensitivity_map, x_obs, y_obs)
    fields = fieldnames(sensitivity_map);
    
    fprintf('\n=== COMPREHENSIVE PARAMETER ANALYSIS ===\n');
    
    % Extract all data
    theta_vals = []; M_vals = []; X_vals = [];
    misfits = []; theta_sens = []; M_sens = []; X_sens = [];
    
    for i = 1:length(fields)
        data = sensitivity_map.(fields{i});
        theta_vals = [theta_vals; data.params(1)];
        M_vals = [M_vals; data.params(2)];
        X_vals = [X_vals; data.params(3)];
        misfits = [misfits; data.misfit];
        theta_sens = [theta_sens; data.theta_sens];
        M_sens = [M_sens; data.M_sens];
        X_sens = [X_sens; data.X_sens];
    end
    
    % Find promising regions (top 25% best fits)
    good_threshold = prctile(misfits, 25);
    good_indices = misfits <= good_threshold;
    
    fprintf('Total test points: %d\n', length(misfits));
    fprintf('Promising solutions (L1 < %.1f): %d points\n', good_threshold, sum(good_indices));
    
    % Promising parameter ranges
    fprintf('\n=== PROMISING PARAMETER RANGES ===\n');
    fprintf('Theta: %.1f to %.1f deg\n', min(theta_vals(good_indices)), max(theta_vals(good_indices)));
    fprintf('M:     %.4f to %.4f\n', min(M_vals(good_indices)), max(M_vals(good_indices)));
    fprintf('X:     %.1f to %.1f\n', min(X_vals(good_indices)), max(X_vals(good_indices)));
    
    % Check negative M values
    negative_M_promising = M_vals(good_indices) < 0;
    if any(negative_M_promising)
        fprintf('-> Promising negative M range: %.4f to %.4f\n', ...
                min(M_vals(good_indices & (M_vals < 0))), ...
                max(M_vals(good_indices & (M_vals < 0))));
    else
        fprintf('-> No promising negative M values found\n');
    end
    
    % Parameter sensitivity ranking
    fprintf('\n=== PARAMETER SENSITIVITY RANKING ===\n');
    sensitivities = [mean(theta_sens(good_indices)), ...
                     mean(M_sens(good_indices)), ...
                     mean(X_sens(good_indices))];
    
    [~, sens_order] = sort(sensitivities, 'descend');
    param_names = {'theta', 'M', 'X'};
    
    for i = 1:3
        idx = sens_order(i);
        fprintf('%d. %s: sensitivity = %.4f\n', i, param_names{idx}, sensitivities(idx));
    end
    
    % Create visualization
    visualize_parameter_space(theta_vals, M_vals, X_vals, misfits, good_indices, x_obs, y_obs);
    
    % Create GA guidance
    create_GA_guidance(theta_vals(good_indices), M_vals(good_indices), X_vals(good_indices), ...
                      theta_sens(good_indices), M_sens(good_indices), X_sens(good_indices), ...
                      sensitivity_map);
end

%% --- Main analysis function
function analyze_parameter_landscape(x_obs, y_obs)
    fprintf('=== COMPREHENSIVE JACOBIAN SENSITIVITY ANALYSIS ===\n');
    
    % Systematic coverage of entire range
    theta_test = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49];
    M_test = [-0.045, -0.035, -0.025, -0.015, -0.005, 0, 0.005, 0.015, 0.025, 0.035, 0.045];
    X_test = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95];
    
    % Latin Hypercube Sampling for gap filling
    n_lhs_samples = 50;
    lhs_samples = get_latin_hypercube_samples(n_lhs_samples);
    
    sensitivity_map = struct();
    test_count = 0;
    
    fprintf('Testing %d systematic points + %d LHS points...\n', ...
            length(theta_test)*length(M_test)*length(X_test), n_lhs_samples);
    
    % Test 1: Systematic grid
    for i = 1:length(theta_test)
        for j = 1:length(M_test)
            for k = 1:length(X_test)
                params = [theta_test(i), M_test(j), X_test(k)];
                
                try
                    [~, J] = jacobian_at_point(params, x_obs, y_obs);
                    misfit_val = calculate_misfit(params, x_obs, y_obs);
                    
                    key = sprintf('sys_%d', test_count);
                    sensitivity_map.(key) = struct(...
                        'params', params, ...
                        'J_norm', norm(J, 'fro'), ...
                        'theta_sens', mean(abs(J(:,1))), ...
                        'M_sens', mean(abs(J(:,2))), ...
                        'X_sens', mean(abs(J(:,3))), ...
                        'misfit', misfit_val, ...
                        'type', 'systematic' ...
                    );
                    test_count = test_count + 1;
                catch
                    fprintf('Skipping point: theta=%.1f, M=%.3f, X=%.1f\n', params);
                end
            end
        end
    end
    
    % Test 2: Latin Hypercube Sampling
    for i = 1:n_lhs_samples
        params = lhs_samples(i, :);
        
        try
            [~, J] = jacobian_at_point(params, x_obs, y_obs);
            misfit_val = calculate_misfit(params, x_obs, y_obs);
            
            key = sprintf('lhs_%d', i);
            sensitivity_map.(key) = struct(...
                'params', params, ...
                'J_norm', norm(J, 'fro'), ...
                'theta_sens', mean(abs(J(:,1))), ...
                'M_sens', mean(abs(J(:,2))), ...
                'X_sens', mean(abs(J(:,3))), ...
                'misfit', misfit_val, ...
                'type', 'LHS' ...
            );
        catch
            fprintf('Skipping LHS point %d\n', i);
        end
    end
    
    fprintf('Successfully analyzed %d parameter combinations\n', length(fieldnames(sensitivity_map)));
    analyze_sensitivity_results(sensitivity_map, x_obs, y_obs);
end

%% --- RUN THE ANALYSIS
analyze_parameter_landscape(x_obs, y_obs);