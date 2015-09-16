function [ log_evidence ] = evidence_2_param_GP(x,p)
    sigma_n = p;
    error_function = @(x,x2) sigma_n^2*(x==x2); 
    parametrized_kernel_function = @(x,x2,p_sigma_f,p_l) p_sigma_f^2*exp((x-x2)^2/(-2*p_l^2));
    X_o = [-1.5 -1 -0.75 -0.4 -0.3 0]';
    Y_o = [-1.6 -1.3 -0.5 0 0.3 0.6]';
    K = zeros(length(X_o));
    for i=1:length(X_o)
        for j=1:length(X_o)
            K(i,j)=parametrized_kernel_function(X_o(i),X_o(j), ...
                x(1),x(2))+error_function(X_o(i),X_o(j));
        end
    end
    log_evidence=Y_o'/K*Y_o+log(det(K));%this is the negative of the log evidence,
                                        %so this is to be minimized
end