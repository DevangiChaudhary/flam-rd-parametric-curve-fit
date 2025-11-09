clear; clc; close all;

%% ---- Parameter ranges ----
theta = linspace(0, 50, 1000);              % degrees
t     = linspace(6, 60, 1500);              % time / parameter
M_vals = [-0.05, 0, 0.01, 0.02, 0.05];          % several M values to compare

%% ---- Component calculations ----
cos_theta = cosd(theta);
sin_theta = sind(theta);

sin_03t = sin(0.3 * t);

% For exp(-M t) we need a matrix t-by-M
[TT, MM] = meshgrid(t, M_vals);
exp_term = exp(-MM .* TT);

%% ---- Plotting ----
figure('Name', 'Component Plots', 'NumberTitle', 'off');

% 1) cos(theta)
subplot(2,2,1)
plot(theta, cos_theta, 'LineWidth', 2)
title('cos(\theta)')
xlabel('\theta (degrees)')
ylabel('cos(\theta)')
grid on

% 2) sin(theta)
subplot(2,2,2)
plot(theta, sin_theta, 'LineWidth', 2)
title('sin(\theta)')
xlabel('\theta (degrees)')
ylabel('sin(\theta)')
grid on

% 3) exp(-M t) for different M values
subplot(2,2,3)
plot(t, exp_term, 'LineWidth', 2)
legendStrings = "M = " + string(M_vals);
legend(legendStrings, 'Location', 'northeast')
title('e^{-Mt} for different M values')
xlabel('t')
ylabel('e^{-Mt}')
grid on

% 4) sin(0.3t)
subplot(2,2,4)
plot(t, sin_03t, 'LineWidth', 2)
title('sin(0.3t)')
xlabel('t')
ylabel('sin(0.3t)')
grid on