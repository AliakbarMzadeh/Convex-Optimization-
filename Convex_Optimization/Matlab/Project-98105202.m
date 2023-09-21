%% Part 1
clear;
clc;

n=863;

A = readtable('svm_train'); % Read Data
A = A{:,:};

Y = A(:,3);
X = [A(:,1),A(:,2)];
diagY = diag(Y);
Xt=rand(n,n);        % Kernel Matrix


one = ones(1,n);

optvalue = rand(3,16);
optalfa = rand(n,16);

k=1;
for C = [0.01, 0.1, 0.5, 1]
    for lambda = [10, 50, 100, 500]
        if lambda==500 && C==0.1
            tic;
        end
        for i=1:1:n
            for j=1:1:n
                Xt(i,j) = exp(-lambda*((X(i,1)-X(j,1))^2 + (X(i,2)-X(j,2))^2));   % Kernel Matrix
            end 
        end
        
        cvx_begin                          % slove problem
            variables alfa(n)
            maximize ( (-1/2)*alfa'*diagY*Xt*diagY*alfa + one*alfa )
            subject to
               alfa'*Y == 0;
               0 <= alfa;
               alfa <= C;
        cvx_end
        
        if lambda==500 && C==0.1
            cvx_time = toc          % detect time (Part 6)
        end
        
        optvalue(1,k) = lambda ;     % Save Data
        optvalue(2,k) = C ;
        optvalue(3,k) = cvx_optval ;
        
        optalfa(:,k) = alfa;
        
        k=k+1;
    end
end

T = array2table( optvalue ) ;     % Write optimal value in .txt 
writetable(T,'optvalue.txt');

T = array2table( optalfa )  ;     % Write optimal alfa in .txt 
writetable(T,'optalfa.txt') ;

%% Part 2
clc;

n=863;

A = readtable('svm_train');    % Read Data
A = A{:,:};

optvalue = readtable('optvalue');
optvalue = optvalue{:,:};
optalfa = readtable('optalfa');
optalfa = optalfa{:,:};


Y = A(:,3);
X = [A(:,1),A(:,2)];
diagY = diag(Y);
Xt=rand(n,n);      % Kernel Matrix



for s=1:1:16
    
    lambda = optvalue(1,s);
    
    for i=1:1:n
        for j=1:1:n
            Xt(i,j) = exp(-lambda*((X(i,1)-X(j,1))^2 + (X(i,2)-X(j,2))^2));  % Kernel Matrix
        end 
    end
    
    b = Y(2,1) - optalfa(:,s)'*diagY*Xt(:,2);     % determine b  
    
    figure;
    subplot(1,2,1);

    for i=1:n                     % plot first Data

        if Y(i) == 1 
            plot(X(i,1),X(i,2),'r.','LineWidth',20,'MarkerSize',16);
        end

        if Y(i) == -1 
            plot(X(i,1),X(i,2),'b.','LineWidth',6,'MarkerSize',16);
        end

        hold on

    end
    title(['lambda = ',num2str(optvalue(1,s))]);
    axis([0 1 0.35 1]);
    subplot(1,2,2);
    for i=1:n                 % plot first Data

        if Y(i) == 1              
            plot(X(i,1),X(i,2),'r.','LineWidth',20,'MarkerSize',16);
        end

        if Y(i) == -1 
            plot(X(i,1),X(i,2),'b.','LineWidth',6,'MarkerSize',16);
        end

        hold on

    end

    for i=0:0.01:1           % plot some point
        for j=0.35:0.01:1
            f =0;
            for k=1:1:n
                f = f + optalfa(k,s)*Y(k,1)*exp(-lambda*((X(k,1)-i)^2 + (X(k,2)-j)^2));
            end 
            if sign( f+b ) == 1                           % decision condition
                plot(i,j,'r.','LineWidth',8,'MarkerSize',8);
            end
            if sign( f+b ) == -1
                plot(i,j,'b.','LineWidth',8,'MarkerSize',8);
            end
            if  f+b  == 0
                plot(i,j,'y.','LineWidth',8,'MarkerSize',8);
            end
        end 
    end
    title(['C = ',num2str(optvalue(2,s))]);
    axis([0 1 0.35 1]);
end

%% Part 3
clc;

n=863;

A = readtable('svm_train'); % Read Data
A = A{:,:};

optvalue = readtable('optvalue');   % Read Data
optvalue = optvalue{:,:};
optalfa = readtable('optalfa');
optalfa = optalfa{:,:};


Y = A(:,3);
X = [A(:,1),A(:,2)];
diagY = diag(Y);
Xt=rand(n,n);        % Kernel

nSupVec = rand(1 , 16);

for s=1:1:16
    
    figure;
    subplot(1,2,1);

    for i=1:n

        if Y(i) == 1 
            plot(X(i,1),X(i,2),'r.','LineWidth',20,'MarkerSize',13);
        end

        if Y(i) == -1 
            plot(X(i,1),X(i,2),'b.','LineWidth',6,'MarkerSize',13);
        end

        hold on

    end
    title(['lambda = ',num2str(optvalue(1,s))]);
   
    subplot(1,2,2);
    nSupVec(1,s) = 0;  % number of support vectors
    for j=1:1:n
        
        if (optalfa(j,s) > exp(-5)) && (optalfa(j,s) < optvalue(2,s) )      % Condition of support vectors
            
            nSupVec(1,s) = nSupVec(1,s) + 1;
            if Y(j,1) == 1 
                plot(X(j,1),X(j,2),'r.','LineWidth',20,'MarkerSize',13);
            end

            if Y(j,1) == -1 
                plot(X(j,1),X(j,2),'b.','LineWidth',6,'MarkerSize',13);
            end            
        end


        hold on

    end
    title(['C = ',num2str(optvalue(2,s))]);
    xlabel(['number of support vectors = ',num2str(nSupVec(1,s))]);
    
end



%% Part 5,6

clc;

n=863;

A = readtable('svm_train');    % Read Data 
A = A{:,:};


Y = A(:,3);
X = [A(:,1),A(:,2)];
diagY = diag(Y);
one = ones(1,863);

Xt=rand(n,n);    % Kernel Matrix



lambda = 500;    % Determine constraints
C = 0.1;
mu = 10^(-4) ;
alpa = 0.01 ;
beta = 0.5 ;
m = 2 * n ;
eta = 0.4 ;
t_int = 40 ;

for i=1:1:863
    for j=1:1:863
        Xt(i,j) = exp(-lambda*((X(i,1)-X(j,1))^2 + (X(i,2)-X(j,2))^2));    % Kernel Matrix
    end 
end



alfa = zeros(n,1);     % optimization variable
for i=1:863                    % Determine initial alfa (optimization variable)
        if Y(i) == 1 
            alfa(i,1) = 0.039;
        end

        if Y(i) == -1 
            alfa(i,1) = 0.049;
        end
end
alfa(1,1) = 0.0625 ;
alfa(2,1) = 0.0625 ;

a=1;

f=0;
f2=0;
tic;
while mu*m > 10^(-8)      % Barrier condition

    while  abs(f2-f) > 0.0001 | a==1
        
        a=0;

        gradg = -1*diagY*Xt*diagY*alfa + one' ;   % Determine Gradient g
        grad2g = -1*diagY*Xt*diagY ;              % Determine Hessian matrix g

        gradfi = zeros(n,1);
        grad2fi = zeros(n,n);

        for i = 1:n
            gradfi(i,1) = (-1/(alfa(i,1)-C))+(-1/alfa(i,1));          % Determine Gradient fi
        end

        for i = 1:n
            grad2fi(i,i) = (1/((alfa(i,1)-C)^2))+(1/(alfa(i,1)^2));   % Determine Hessian matrix fi
        end

        gradf = gradg + mu*gradfi ;       % Determine Gradient f  (Optimization function)
        grad2f = grad2g + mu*grad2fi ;    % Determine Hessian matrix f    f(alfa) = g(alfa)+ mu*fi(alfa)

        NewtonMatrix = cat(2,grad2f,Y) ;     % Making Newton matrix [hessian f , Y ; Y' , [0]]
        R = cat(2,Y',[0]);
        NewtonMatrix = cat(1,NewtonMatrix,R);

        Newton2 = cat(1,-gradf,[0]);         % Making Newton vector [grad f , [0]]

        V = inv(NewtonMatrix)*Newton2;       % Determine V of Newton's method 
        V(n+1,:)=[] ; 

        t = t_int;                          % t = initial t

        g2 = (-1/2)*(alfa+ t*V)'*diagY*Xt*diagY*(alfa+ t*V) + one*(alfa+ t*V) ;
        fi2 = 0 ;
        for i = 1:n
            fi2 = fi2 -log(alfa(i,1)+ t*V(i,1)) -log(C-alfa(i,1)-t*V(i,1)) ;
        end
        f2 = g2 + mu*fi2 ;        % f( alfa + t*V)
        
        while imag(f2) ~= 0 
            t = beta*t;                            

            g2 = (-1/2)*(alfa+ t*V)'*diagY*Xt*diagY*(alfa+ t*V) + one*(alfa+ t*V) ;
            fi2 = 0 ;
            for i = 1:n
                fi2 = fi2 -log(alfa(i,1)+ t*V(i,1)) -log(C-alfa(i,1)-t*V(i,1)) ;
            end
            f2 = g2 + mu*fi2 ;
            
                       
        end

        g = (-1/2)*alfa'*diagY*Xt*diagY*alfa + one*alfa ;
        g2 = (-1/2)*(alfa+ t*V)'*diagY*Xt*diagY*(alfa+ t*V) + one*(alfa+ t*V) ;
        fi = 0 ;
        fi2 = 0 ;
        for i = 1:n
            fi = fi -log(alfa(i,1)) -log(C-alfa(i,1)) ;
            fi2 = fi2 -log(alfa(i,1)+ t*V(i,1)) -log(C-alfa(i,1)-t*V(i,1)) ;
        end
        f = g + mu*fi ;           % f(alfa)
        f2 = g2 + mu*fi2 ;        % f( alfa + t*V)       
        
        while f2 > f + alpa * t * gradf' * V         % Backtracking line search condition
            t = beta*t;                              % t = beta*t
            
            if t<0.0001
                break;
            end           
            
            g2 = (-1/2)*(alfa+ t*V)'*diagY*Xt*diagY*(alfa+ t*V) + one*(alfa+ t*V) ;
            fi2 = 0 ;
            for i = 1:n
                fi2 = fi2 -log(alfa(i,1)+ t*V(i,1)) -log(C-alfa(i,1)-t*V(i,1)) ;
            end
            f2 = g2 + mu*fi2 ;      
        end
 
        alfa = alfa + t*V   ;   % update to next step

    end
    
    mu = mu * eta ;     % Barrier step
end

algorithm_time = toc       % Determine algorithm time 


b = Y(1,1) - alfa'*diagY*Xt(:,1);

    figure;              % plot Final result
    subplot(1,2,1);

    for i=1:n

        if Y(i) == 1 
            plot(X(i,1),X(i,2),'r.','LineWidth',20,'MarkerSize',16);
        end

        if Y(i) == -1 
            plot(X(i,1),X(i,2),'b.','LineWidth',6,'MarkerSize',16);
        end

        hold on

    end
    title(['lambda = ',num2str(500)]);
    axis([0 1 0.35 1]);
    subplot(1,2,2);
    for i=1:n

        if Y(i) == 1 
            plot(X(i,1),X(i,2),'r.','LineWidth',20,'MarkerSize',16);
        end

        if Y(i) == -1 
            plot(X(i,1),X(i,2),'b.','LineWidth',6,'MarkerSize',16);
        end

        hold on

    end

    for i=0:0.01:1
        for j=0.35:0.01:1
            f =0;
            for r=1:1:n
                f = f + alfa(r,1)*Y(r,1)*exp(-lambda*((X(r,1)-i)^2 + (X(r,2)-j)^2));
            end 
            if sign( f+b ) == 1
                plot(i,j,'r.','LineWidth',8,'MarkerSize',8);
            end
            if sign( f+b ) == -1
                plot(i,j,'b.','LineWidth',8,'MarkerSize',8);
            end
            
        end 
    end
    title(['C = ',num2str(0.1)]);
    axis([0 1 0.35 1]);

    
    
    figure;         % Detemine and plot support vectors
    subplot(1,2,1);

    for i=1:n

        if Y(i) == 1 
            plot(X(i,1),X(i,2),'r.','LineWidth',20,'MarkerSize',13);
        end

        if Y(i) == -1 
            plot(X(i,1),X(i,2),'b.','LineWidth',6,'MarkerSize',13);
        end

        hold on

    end
    title(['lambda = ',num2str(lambda)]);
   
    subplot(1,2,2);
    nSupVec = 0;
    for j=1:1:n
        
        if (alfa(j,1) > exp(-5)) && (alfa(j,1) < C )
            
            nSupVec = nSupVec + 1;
            if Y(j,1) == 1 
                plot(X(j,1),X(j,2),'r.','LineWidth',20,'MarkerSize',13);
            end

            if Y(j,1) == -1 
                plot(X(j,1),X(j,2),'b.','LineWidth',6,'MarkerSize',13);
            end            
        end


        hold on

    end
    title(['C = ',num2str(C)]);
    xlabel(['number of support vectors = ',num2str(nSupVec)]);
    



