clc;
clear;
close all;

figure;
x = linspace(-5,5);
y = linspace(-5,5);
[X,Y] = meshgrid(x,y);
Z = (X.^2 + Y - 11).^2 + (X+Y.^2-7).^2;
contourf(X,Y,Z,10);
pause(0.00001);

tic;
%% Problem Definition
CostFunction=@(x) myCostFunction(x);        % Cost Function
nVar=2;            % Number of Decision Variables
VarSize=[1 nVar];   % Size of Decision Variables Matrix
VarMin=-5;         % Lower Bound of Variables
VarMax= 5;         % Upper Bound of Variables
my_epsilon = 0.0000001;

%% PSO Parameters
MaxIt=1000;      % Maximum Number of Iterations
nPop=10;        % Population Size (Swarm Size)
% PSO Parameters
w=1;            % Inertia Weight
wdamp=0.99;     % Inertia Weight Damping Ratio
c1=1.5;         % Personal Learning Coefficient
c2=2.0;         % Global Learning Coefficient
% Velocity Limits
VelMax=0.1*(VarMax-VarMin);
VelMin=-VelMax;

%% Initialization

empty_particle.PreviousPosition=[];
empty_particle.Position=[];
empty_particle.Cost=[];
empty_particle.Velocity=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];

particle=repmat(empty_particle,nPop,1);

GlobalBest.Cost=inf;

costFunctionCounter = 0;
for i=1:nPop
    
    % Initialize Position
    particle(i).Position=unifrnd(VarMin,VarMax,VarSize);
    
    % Initialize Velocity
    particle(i).Velocity=zeros(VarSize);
    
    % Evaluation
    particle(i).Cost=CostFunction(particle(i).Position);
    costFunctionCounter = costFunctionCounter + 1;
    
    % Update Personal Best
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    particle(i).PreviousPosition = particle(i).Position; 
    
    % Update Global Best
    if particle(i).Best.Cost<GlobalBest.Cost
        
        GlobalBest=particle(i).Best;
        
    end
    
end

BestCost=zeros(MaxIt,1);
MeanCost=zeros(MaxIt,1);

%% PSO Main Loop
finalResult = [];
finalResult_f = [-100];
for it=1:MaxIt
    
    for i=1:nPop
        
        %hold on;
        %Marked=[particle(i).PreviousPosition(1) particle(i).PreviousPosition(2)];
        %plot(Marked(:,1),Marked(:,2),'Marker', 'none');
        %pause(0.001);
        
        
        
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
        
        % Apply Velocity Limits
        particle(i).Velocity = max(particle(i).Velocity,VelMin);
        particle(i).Velocity = min(particle(i).Velocity,VelMax);
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity Mirror Effect
        IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);
        
        % Apply Position Limits
        particle(i).Position = max(particle(i).Position,VarMin);
        particle(i).Position = min(particle(i).Position,VarMax);
        
        % Evaluation
        particle(i).Cost = CostFunction(particle(i).Position);
        costFunctionCounter = costFunctionCounter + 1;
        
        hold on;
        Marked=[particle(i).Position(1) particle(i).Position(2)];
        plot(Marked(:,1),Marked(:,2),'*');
        pause(0.001);

        particle(i).PreviousPosition = particle(i).Position;
        
        %% Update Personal Best
        if particle(i).Cost<particle(i).Best.Cost  
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            
            % Update Global Best
            if particle(i).Best.Cost<GlobalBest.Cost
                GlobalBest=particle(i).Best;
                if particle(i).Best.Cost < my_epsilon
                    if any(finalResult_f(:) ~= particle(i).Best.Position(1))
                        finalResult = [finalResult particle(i).Best.Position];
                        finalResult_f = [finalResult_f particle(i).Best.Position(1)];
                    end    
                end
            end
        end
        
        %% Re init
        if particle(i).Best.Cost < my_epsilon
            GlobalBest.Cost=inf;
            for zz=1:nPop
                particle(zz).Position=unifrnd(VarMin,VarMax,VarSize);
                particle(zz).Velocity=zeros(VarSize);
                particle(zz).Cost=CostFunction(particle(zz).Position);
                costFunctionCounter = costFunctionCounter + 1;
                particle(zz).Best.Position=particle(zz).Position;
                particle(zz).Best.Cost=particle(zz).Cost;
                if particle(zz).Best.Cost<GlobalBest.Cost
                    GlobalBest=particle(zz).Best;
                end
            end 
        end
    end
    
    %% Calculate Mean Cost of iteration
    tmpCost = 0;
    for i=1:nPop
        tmpCost = tmpCost + particle(i).Cost;
        MeanCost(it) = tmpCost;
    end
    
    
    BestCost(it)=GlobalBest.Cost;
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it)) ' Best Sol:' num2str(GlobalBest.Position)]);
    w=w*wdamp;
    
    hold on;
    Marked=[GlobalBest.Position(1) GlobalBest.Position(2)];
    plot(Marked(:,1),Marked(:,2),'o');
    pause(0.001);
    
    delete( findobj(gca, 'type', 'line') );

end

BestSol = GlobalBest;


%% Remove Duplicate 
finalResult = round(finalResult,3);
Ac = zeros(size(finalResult));
for k1 = 1:length(finalResult)-1
    Ac(k1) = sum(finalResult(k1) == finalResult(k1+1:end));
end
Output = finalResult(Ac == 0);


toc

%% Results
disp(['Best Points: ']);
reshape(Output,[2,4])

disp(['Number Of Population: ' num2str(nPop)]);
disp(['Number Of Iteration: ' num2str(MaxIt)]);
disp(['Number Of Cost Function Call: ' num2str(costFunctionCounter)]);


figure;
%plot(BestCost,'LineWidth',2);
semilogy(BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;

figure;
%plot(BestCost,'LineWidth',2);
semilogy(MeanCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Mean Cost');
grid on;

