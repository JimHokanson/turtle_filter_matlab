function test_speed_02()
%

%{
x = rand(1,1e7);
[b,a] = cheby1(6,3,2000/100000);
y3 = turtle.filter(b,1,x);
%}

MIN_ORDER = 1;
MAX_ORDER = 10;
N = 10;
data_length = 1e7;
x = rand(1,data_length);
times = zeros(N,3,MAX_ORDER);
for i = 1:N
    for j = MIN_ORDER:MAX_ORDER
        [b,a] = cheby1(j,3,2000/100000);
        
        fprintf('Iteration %d for order %d\n',i,j);
        tic
        y1 = filter(b,1,x);
        times(i,1,j) = toc;
        
        %     tic
        %     y2 = sl.array.mex_filter(b,1,x);
        %     times(i,2,j) = toc;
        
        tic
        y3 = turtle.filter(b,1,x);
        times(i,3,j) = toc;
    end
end

t1 = squeeze(median(times(:,1,:)));
% t2 = squeeze(median(times(:,2,:)));
t3 = squeeze(median(times(:,3,:)));

subplot(1,2,1)
plot(1000*t1)
hold on
plot(1000*t3)
hold off
ylabel(sprintf('Execution times (ms)'))
title(sprintf('FIR for data length %d',data_length))
legend({'ML','Me'})
subplot(1,2,2)
plot(t1./t3)
ylabel('Speedup, Me/Ml')

keyboard

end