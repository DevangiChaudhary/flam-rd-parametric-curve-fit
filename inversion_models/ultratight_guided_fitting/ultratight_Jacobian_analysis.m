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
    plot([prctile(misfits, 10), prctile(misfits, 10)], yl, 'r--', 'linewidth', 2);
    plot([prctile(misfits, 25), prctile(misfits, 25)], yl, 'g--', 'linewidth', 2);
    xlabel('L1 Misfit'); ylabel('Frequency');
    title('Misfit Distribution');
    legend('All points', '10th percentile', '25th percentile');
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
    
    sgtitle('ULTRA-TIGHT Parameter Space Analysis');
end

%% --- Create ULTRA-TIGHT GA guidance
function create_GA_guidance(theta_vals, M_vals, X_vals, theta_sens, M_sens, X_sens, sensitivity_map)
    fprintf('\n=== ULTRA-TIGHT GA OPTIMIZATION GUIDANCE ===\n');
    
    % Use TOP 10% best solutions instead of 25% for TIGHTER ranges
    excellent_threshold = prctile([theta_vals, M_vals, X_vals, theta_sens, M_sens, X_sens], 10);
    excellent_indices = [theta_vals, M_vals, X_vals, theta_sens, M_sens, X_sens] <= excellent_threshold;
    
    if sum(excellent_indices) < 10
        fprintf('Not enough excellent solutions. Using top 15%% instead.\n');
        excellent_threshold = prctile([theta_vals, M_vals, X_vals, theta_sens, M_sens, X_sens], 15);
        excellent_indices = [theta_vals, M_vals, X_vals, theta_sens, M_sens, X_sens] <= excellent_threshold;
    end
    
    % ULTRA-TIGHT ranges (mean ± 1.0*std instead of 1.5*std)
    theta_range = [max(0, mean(theta_vals) - 1.0*std(theta_vals)), ...
                   min(50, mean(theta_vals) + 1.0*std(theta_vals))];
    M_range = [max(-0.05, mean(M_vals) - 1.0*std(M_vals)), ...
               min(0.05, mean(M_vals) + 1.0*std(M_vals))];
    X_range = [max(0, mean(X_vals) - 1.0*std(X_vals)), ...
               min(100, mean(X_vals) + 1.0*std(X_vals))];
    
    % Ensure minimum range sizes to prevent overly restrictive bounds
    min_ranges = [5, 0.01, 10]; % min theta range, min M range, min X range
    
    if diff(theta_range) < min_ranges(1)
        center = mean(theta_range);
        theta_range = [max(0, center - min_ranges(1)/2), min(50, center + min_ranges(1)/2)];
    end
    
    if diff(M_range) < min_ranges(2)
        center = mean(M_range);
        M_range = [max(-0.05, center - min_ranges(2)/2), min(0.05, center + min_ranges(2)/2)];
    end
    
    if diff(X_range) < min_ranges(3)
        center = mean(X_range);
        X_range = [max(0, center - min_ranges(3)/2), min(100, center + min_ranges(3)/2)];
    end
    
    fprintf('ULTRA-TIGHT GA search ranges:\n');
    fprintf('theta: [%.1f, %.1f] deg (range: %.1f)\n', theta_range, diff(theta_range));
    fprintf('M:     [%.4f, %.4f] (range: %.4f)\n', M_range, diff(M_range));
    fprintf('X:     [%.1f, %.1f] (range: %.1f)\n', X_range, diff(X_range));
    
    % Calculate search volume reduction
    full_volume = 50 * 0.1 * 100;
    tight_volume = diff(theta_range) * diff(M_range) * diff(X_range);
    reduction = (1 - tight_volume / full_volume) * 100;
    fprintf('Search volume reduced by %.1f%%\n', reduction);
    
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
    
    % Save ULTRA-TIGHT guidance
    guidance = struct();
    guidance.theta_range = theta_range;
    guidance.M_range = M_range;
    guidance.X_range = X_range;
    guidance.focus_parameter = focus_param;
    guidance.best_start = best_start;
    guidance.best_start_misfit = best_misfit;
    
    save('ga_guidance_ultra_tight.mat', 'guidance');
    fprintf('\nULTRA-TIGHT guidance saved to ga_guidance_ultra_tight.mat\n');
end

%% --- Analyze results with TIGHTER criteria
function analyze_sensitivity_results(sensitivity_map, x_obs, y_obs)
    fields = fieldnames(sensitivity_map);
    
    fprintf('\n=== ULTRA-TIGHT PARAMETER ANALYSIS ===\n');
    
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
    
    % Find EXCELLENT regions (top 10% best fits - TIGHTER)
    excellent_threshold = prctile(misfits, 10);
    excellent_indices = misfits <= excellent_threshold;
    
    fprintf('Total test points: %d\n', length(misfits));
    fprintf('EXCELLENT solutions (L1 < %.1f): %d points\n', excellent_threshold, sum(excellent_indices));
    
    % If too few excellent solutions, use top 15%
    if sum(excellent_indices) < 10
        fprintf('Too few excellent solutions. Using top 15%% instead.\n');
        excellent_threshold = prctile(misfits, 15);
        excellent_indices = misfits <= excellent_threshold;
    end
    
    % EXCELLENT parameter ranges (TIGHTER)
    fprintf('\n=== EXCELLENT PARAMETER RANGES ===\n');
    fprintf('Theta: %.1f to %.1f deg\n', min(theta_vals(excellent_indices)), max(theta_vals(excellent_indices)));
    fprintf('M:     %.4f to %.4f\n', min(M_vals(excellent_indices)), max(M_vals(excellent_indices)));
    fprintf('X:     %.1f to %.1f\n', min(X_vals(excellent_indices)), max(X_vals(excellent_indices)));
    
    % Check negative M values
    negative_M_excellent = M_vals(excellent_indices) < 0;
    if any(negative_M_excellent)
        fprintf('-> Excellent negative M range: %.4f to %.4f\n', ...
                min(M_vals(excellent_indices & (M_vals < 0))), ...
                max(M_vals(excellent_indices & (M_vals < 0))));
    else
        fprintf('-> No excellent negative M values found\n');
    end
    
    % Parameter sensitivity ranking
    fprintf('\n=== PARAMETER SENSITIVITY RANKING ===\n');
    sensitivities = [mean(theta_sens(excellent_indices)), ...
                     mean(M_sens(excellent_indices)), ...
                     mean(X_sens(excellent_indices))];
    
    [~, sens_order] = sort(sensitivities, 'descend');
    param_names = {'theta', 'M', 'X'};
    
    for i = 1:3
        idx = sens_order(i);
        fprintf('%d. %s: sensitivity = %.4f\n', i, param_names{idx}, sensitivities(idx));
    end
    
    % Create visualization
    visualize_parameter_space(theta_vals, M_vals, X_vals, misfits, excellent_indices, x_obs, y_obs);
    
    % Create ULTRA-TIGHT GA guidance
    create_GA_guidance(theta_vals(excellent_indices), M_vals(excellent_indices), X_vals(excellent_indices), ...
                      theta_sens(excellent_indices), M_sens(excellent_indices), X_sens(excellent_indices), ...
                      sensitivity_map);
end

%% --- Enhanced main analysis function with DENSER sampling
function analyze_parameter_landscape(x_obs, y_obs)
    fprintf('=== ULTRA-TIGHT JACOBIAN SENSITIVITY ANALYSIS ===\n');
    
    % DENSER systematic coverage for better precision
    theta_test = [15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45];
    M_test = [-0.01, -0.005, 0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035];
    X_test = [45, 48, 50, 52, 55, 58, 60, 62, 65, 68, 70, 72, 75];
    
    % More LHS samples for better coverage
    n_lhs_samples = 80;
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
                    continue;
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
            continue;
        end
    end
    
    fprintf('Successfully analyzed %d parameter combinations\n', length(fieldnames(sensitivity_map)));
    analyze_sensitivity_results(sensitivity_map, x_obs, y_obs);
end

%% --- RUN THE ULTRA-TIGHT ANALYSIS
analyze_parameter_landscape(x_obs, y_obs);