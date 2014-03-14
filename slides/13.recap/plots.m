function plots(kappa, gamma, N, maxP)

P=1:maxP;
clf;
plot(P, 1./(1./P+2*(kappa+gamma*N*8)),'r-','linewidth',2);
hold on
plot(P, 1./(1./P+4*(kappa+gamma*8*N./sqrt(P))),'g-','linewidth',2);
plot(P, 1./(1./P+kappa+gamma*8*N^2),'b-','linewidth',2);
plot(P,P,'k-','linewidth',2);

legend('Strip DD','Block DD','Matrix rows','Ideal')
