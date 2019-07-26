%%Dependencies: https://git.dwavesys.local/users/jraymond/repos/utilities/browse
%%              https://git.dwavesys.local/users/jraymond/repos/orang_modified/browse
%%              https://git.dwavesys.local/projects/APPS/repos/orang/browse
%%              https://git.dwavesys.local/users/jraymond/repos/process_samples/browse
%%              matlab package for QPU
%%These all have sufficient analogues in Ocean (python) and some of the other
%%functions expressed (such as tiling) are also simplified.
clearvars
close all
 
%%Set up QPU parameters, mostly defaults %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nProgrammings = 10;
numReads = 1000;
learningRate = 0.25;%0.5 is fine for RAN-Inf
%Generate 16 RAN-Inf problems
qpuName{1}='BAY1_X_INTERNAL';
qpuName{2}='DW_2000Q_5';
%Set up two equally sized BAY1_X_INTERNAL with defaults:
fileName= ['sapiMetaParameters.mat'];
if exist(fileName,'file')
    load(fileName,'sapiMetaParameters');
end
for hwI=1:numel(qpuName)
    sapiMetaParameters{hwI}.url='https://cloud.dwavesys.com/sapi';
    sapiMetaParameters{hwI}.solverName=qpuName{hwI};
    sapiMetaParameters{hwI}.token='DWV-8ff55522332bef0647568359209df718d2ed9355';
    try
        sapiMetaParameters{hwI}=sapiWrapper(sapiMetaParameters{hwI});
    catch e
        if ~exist(fileName,'file')
            error('sapi unreachable');
        else
            warning('sapi unreachable ');
        end
    end
    sapiMetaParameters{hwI}.sapiParameters.num_reads = numReads;
    nVar = sapiMetaParameters{hwI}.remoteSolver.property.num_qubits;
    nEdges = size(sapiMetaParameters{hwI}.remoteSolver.property.couplers,2);
    r=1+sapiMetaParameters{hwI}.remoteSolver.property.couplers(1,:);
    c=1+sapiMetaParameters{hwI}.remoteSolver.property.couplers(2,:);
    hwMask{hwI} = sparse(r,c,1,nVar,nVar);
end
%%Set up QPU parameters, mostly defaults %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%%Set up Intersection graphs (multiple QPU, same problem set) %%%%%%%%%%%%%
%% In this example each tile is different, but needn't be so  %%%%%%%%%%%%%
[r,c]=find(hwMask{1}.*hwMask{2});
nEdges = length(r);
assert( sapiMetaParameters{1}.remoteSolver.property.num_qubits==sapiMetaParameters{2}.remoteSolver.property.num_qubits);
%%Set up Intersection graphs (multiple QPU, same problem set) %%%%%%%%%%%%%
 
 
%%Some parameters for the exact solver (orang), ocean has equivalent %%%%%%
L = 4;
N = sqrt(nVar/(2*L));
M = N;
assert(M==16,'Code is specific to 16 by 16 in this example');
assert(nVar==N*M*L*2);
varOrder = chimeraVarOrder(M,N,L);
%%Some parameters for the exact solver (orang), ocean has equivalent %%%%%%
 
%%%%%%%%%%% Create a tiling (forbid edges spanning C4 blocks %%%%%%%%%%%%%%
tileScale=4;
rIntra=r(c-r<8);
cIntra=c(c-r<8);
rInterCol=r(c-r==8);
cInterCol=c(c-r==8);
rInterRow=r(c-r==8*N);
cInterRow=c(c-r==8*N);
rowIndex=floor((0:nVar-1)/(8*N));
colIndex=mod(floor((0:nVar-1)/8),N);
validRow=mod(rowIndex(cInterRow),tileScale)~=0;
validCol=mod(colIndex(cInterCol),tileScale)~=0;
adjMasked=sparse(rInterCol(validCol),cInterCol(validCol),1,nVar,nVar) + ...
    sparse(rInterRow(validRow),cInterRow(validRow),1,nVar,nVar) + ...
    sparse(rIntra,cIntra,1,nVar,nVar);
%%%%%%%%%%% Create a tiling (forbid edges spanning C4 blocks %%%%%%%%%%%%%%
 
%%%%%%%%%%% Generate problems in this case so called RANInf %%%%%%%%%%%%%%%
 
if ~exist(fileName,'file')
    %Turn into a RAN-inf spin glass
    JFull = adjMasked.*sparse(r,c,sign(1-2*rand(length(r),1)),nVar,nVar);
    for hwI=1:numel(qpuName)
        DrawChimeraProblem(hwMask{hwI}, zeros(nVar,1), JFull, ['J_' qpuName{hwI} '.eps'], [M,N,4], false);
    end
    save(fileName,'sapiMetaParameters','JFull');
else
    load(fileName,'JFull');
end
hFull = zeros(nVar,1);
activeQubits=find(any(JFull+JFull'));
%%%%%%%%%%% Generate problems in this case so called RANInf %%%%%%%%%%%%%%%
 
%%%%%%%%%%% Collect and save QPU data                       %%%%%%%%%%%%%%%
for hwI=1:numel(qpuName)
%Collect data, 10 different gauges:
fileName= ['sampleSet_' qpuName{hwI} '.mat'];
if ~exist(fileName,'file')
    spinReversals = 3-2*randi(2,nVar,nProgrammings);
    samples = zeros(nVar,numReads,nProgrammings);
    for i=1:nProgrammings
        %Draw 1000 samples (manual specification of the spin-reversal
        %transformation per programming)
        sapiMetaParameters{hwI}.spinReversal = spinReversals(:,i);
        [sapiMetaParameters{hwI},answer] = sapiWrapper(sapiMetaParameters{hwI},hFull,JFull);
        %Solutions are reduced here over active qubits already, easier just to
        %bump up:
        samples(activeQubits,:,i) = answer.solutions;
    end
    save(fileName,'samples','spinReversals')
else
    load(fileName);
end
%%%%%%%%%%% Collect and save QPU data                       %%%%%%%%%%%%%%%
 
%%%%%%%%%%% For each tile estimate the distribution temperature %%%%%%%%%%%
for nRows=0:3
    for nCols=0:3
        thisTile = bsxfun(@plus,M*8*(0:3),(1:32)') + 32*nCols + (M*8)*nRows;
         
        %%QPU energies this cell                          %%%%%%%%%%%%%%%%%
        for i=1:nProgrammings
            energies(i,:) = sum(samples(thisTile(:),:,i).*(JFull(thisTile(:),thisTile(:))*samples(thisTile(:),:,i)));
        end
        meanEnQPU = mean(energies(:));
        stdDevEnQPU = sqrt(var(energies(:)));
        stdErrEnQPU = stdDevEnQPU/sqrt(length(energies));%Assumes independent samples
        %stdErrEn2(1+4*nRows+nCols) = ...
        %   sqrt(var(mean(energies,2))/nProgrammings);  %Assumes independent programmings
         
        varOrder0 = intersect(varOrder,thisTile,'stable');
         
         
        %%Maximum-Likelihood estimate by gradient descent %%%%%%%%%%%%%%%%%
        betaSequence = 1; %Could do better, but doesn't matter
        enMean=Inf;
        betaSequence=betaSequence(end);
        while abs(meanEnQPU - enMean)>stdErrEnQPU
            [~,samplesOut] = orang_sample_hJ(betaSequence(end)*hFull(varOrder0),betaSequence(end)*JFull(varOrder0,varOrder0),10^4,0,1:length(varOrder0));
            en = sum(samplesOut.*(JFull(varOrder0,varOrder0)*samplesOut));
            enMean = mean(en);
            enStdDev = sqrt(var(en));
            betaSequence(end+1) = betaSequence(end) - learningRate/max(1,enStdDev)*(meanEnQPU - enMean);
        end
        betaEstML(nRows+1,nCols+1,hwI)=betaSequence(end);
        %%Maximum-Likelihood estimate by gradient descent %%%%%%%%%%%%%%%%%
         
        figure(1)
        plot(betaSequence);
        hold on
        xlabel('Iterations');
        ylabel('Inverse temperature estimate');
         
        %I would suggest also adding the pseudo-likelihood estimator, see
        %equation (10) of Global warming paper:
        betaEstMPL(nRows+1,nCols+1,hwI) = betabybitflip_hJ(samples(thisTile(:),:) ...
            ,hFull(thisTile(:)),JFull(thisTile(:),thisTile(:)));
         
        %I would suggest adding error bars: simplest case, fit to E + sigma
        %and E - sigma. Better is to use a jack-knife estimate over the
        %different programmings
    end
end
 
end
LB=10^min(floor(log10([betaEstMPL(:);betaEstML(:)])));
UB=10^max(ceil(log10([betaEstMPL(:);betaEstML(:)])));
colHW='rb';
betaEstMPL=reshape(betaEstMPL,[],2);
betaEstML=reshape(betaEstML,[],2);
for hwI=1:2
    figure(2)
    plot(betaEstMPL(:,hwI),[colHW(hwI) 'x'],'MarkerSize',10)
    hold on
    ylabel('Inverse temperature (MLPL estimator)');
    xlabel('C4 tile label')
    legend(qpuName{1},qpuName{1},qpuName{2},qpuName{2})
    figure(3)
    plot(betaEstML(:,hwI),[colHW(hwI) '+'],'MarkerSize',10)
    hold on
    xlabel('C4 tile')
    ylabel('Inverse temperature (ML estimator) ');
    xlabel('C4 tile label')
    legend(qpuName{1},qpuName{2});
    figure(5+hwI)
    plot(betaEstML(:,hwI),betaEstMPL(:,hwI),[colHW(hwI) '+'],'MarkerSize',10)
    hold on
    plot([LB,UB],[LB,UB],'k:')
    xlabel('Maximum likelihood');
    ylabel('Maximum Pseudo-likelihood');
    figure(7+hwI)
    heatmap(reshape(betaEstML(:,hwI),4,4));
    xlabel('Chimera tile X');
    ylabel('Chimera tile Y');
end
figure(4)
plot(betaEstMPL(:,1),betaEstMPL(:,2),'x','MarkerSize',10)
hold on
plot([LB,UB],[LB,UB],'k:')
xlabel(qpuName{1});
ylabel(qpuName{2});
title('MLPL temperature estimator');
set(gca,'yscale','log')
set(gca,'xscale','log')
figure(5)
plot(betaEstML(:,1),betaEstML(:,2),'x','MarkerSize',10);
hold on
plot([LB,UB],[LB,UB],'k:')
set(gca,'yscale','log')
set(gca,'xscale','log')
xlabel(qpuName{1});
ylabel(qpuName{2});
%%%%%%%%%%% For each tile estimate the distribution temperature %%%%%%%%%%%
disp(betaEstMPL)
disp(betaEstML);
title('Maximum Likelihood estimator');