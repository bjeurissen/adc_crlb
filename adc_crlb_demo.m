function adc_crlb_demo()
    % Calculates the CRLB for ADC as a function of the dependent variables in x, and independent variables b and sigma
    function val = cost(x,b,sigma)
        if size(unique(b),1) > 1
            df = @(x,b) [exp(-b*x(2)) -b.*x(1).*exp(-b*x(2))]; % partial derivates of ADC model with respect to S_0 (x(1)) and ADC (x(2))
            dy = df(x,b);
            I = (dy'*dy)./sigma^2; % Fisher information matrix (FIM)
            invI = inv(I+eps); % Inverse of the FIM
            val = invI(2,2); % Second diagonal element corresponds to the lower bound on the variance of ADC (x(2)) 
        else
            val = Inf;
        end
    end

options=optimoptions('fmincon','OptimalityTolerance',1e-12,'StepTolerance',1e-20,'Display','none');%,'MaxFunctionEvaluations',inf,'MaxIterations',3000);
sigma=1; % can be set to any value, as it does not affect the optimum

low_b = 0.0; % mininally achievable b-value

% What is the optimal b-value for a given ADC?
adcs = 0.5:0.5:2;
bmaxes = low_b+0.01:0.01:3;
figure; hold all;
for i = 1:size(adcs,2)
    c = zeros(size(bmaxes));
    for j = 1:size(bmaxes,2)
        c(j) = cost([1;adcs(i)],[low_b; bmaxes(j)],sigma);
    end
    [b_opt, val_opt] = fmincon(@(b)cost([1;adcs(i)],[low_b; b],sigma),low_b+1.1/adcs(i),[],[],[],[],low_b,[],[],options);
    h = plot(bmaxes,sqrt(val_opt)./sqrt(c),'LineWidth',2); line_color = get(h,'Color');
    h = scatter(max(b_opt),1,50,line_color,'filled'); set(h,'HandleVisibility','off');
end
xlabel('b (ms/μm^2)'); ylabel('SNR relative to optimum');
legend(arrayfun(@(x) ['ADC = ' num2str(x) ' μm^2/ms'],adcs,'UniformOutput',false))


% What is the optimal b-value as a function of ADC?
adcs = 0.1:0.01:1.5;
b_opts = zeros(size(adcs));
for i = 1:size(adcs,2)
    b_opts(i) = fmincon(@(b)cost([1;adcs(i)],[low_b; b],sigma),low_b+1.1/adcs(i),[],[],[],[],low_b,[],[],options);
end
figure;
plot(adcs,b_opts,'LineWidth',2); hold all;
plot(adcs,low_b+1.1./adcs,'LineWidth',2);
xlim([0 max(adcs)]); xlabel('ADC'); ylabel('b_{opt}');
legend('optimal','low\_b+1.1/ADC heuristic')


% What is the optimal proportion of low b-value samples?
adc = 1;
bmax = 1/adc;
figure; hold all;
h = line([0 1],[1 1],'LineWidth',1,'LineStyle','--','Color',[0.5 0.5 0.5]); set(h,'HandleVisibility','off')
ns = [3:9 10:10:100];
for i = 1:size(ns,2)
    n = ns(i);
    nb0s = 1:n-1;
    c = zeros(size(nb0s));
    for j = 1:size(nb0s,2)
        nb0 = nb0s(j);
        c(j) = cost([1;adc],[ones([nb0 1])*low_b; ones([n-nb0 1])*bmax],sigma);
    end
    h = plot(nb0s./n,sqrt(c(1))./sqrt(c),'LineWidth',2); line_color = get(h,'Color');
    [~,idx] = min(c);
    h = scatter(nb0s(idx)./n,sqrt(c(1))./sqrt(c(idx)),50,line_color,'filled'); set(h,'HandleVisibility','off');
end
xlabel(['proportion of b = ' num2str(low_b) ' ms/μm^2 scans']); ylabel(['SNR gain relative to using 1 b = ' num2str(low_b) ' ms/μm^2 scan']);
legend(arrayfun(@(x) ['n = ' num2str(x)],ns,'UniformOutput',false))

end
