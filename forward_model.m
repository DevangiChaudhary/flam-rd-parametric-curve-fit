theta = 30.0438;
M = 0.02999; 
X = 55.0156;
t = linspace(6, 60, 1500);  % 1500 points from t=6 to t=60

xf = t.*cosd(theta) - exp(M*abs(t)) .* sin(0.3*t) .* sind(theta) + X;
yf = 42 + t.*sind(theta) + exp(M*abs(t)) .* sin(0.3*t) .* cosd(theta);

% Scatter plot xf vs t
figure;
subplot(2,1,1);
scatter(t, xf, 10, 'b', 'filled');  % 10 is marker size
xlabel('t');
ylabel('x');
title(['x vs t (\theta = ' num2str(theta) ', M = ' num2str(M) ', X = ' num2str(X) ')']);
grid on;

% Scatter plot yf vs t
subplot(2,1,2);
scatter(t, yf, 10, 'r', 'filled');
xlabel('t');
ylabel('y');
title(['y vs t (\theta = ' num2str(theta) ', M = ' num2str(M) ', X = ' num2str(X) ')']);
grid on;

figure;
scatter(xf, yf, 'filled');
xlabel('x');
ylabel('y');
title('y vs x');
grid on;