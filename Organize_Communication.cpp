#include<iostream>
#include<iomanip>
using namespace std;
#define nt 8
#define nc 13
typedef struct node{     //此处定义一个链表
	int data;
	struct node *next;
}LinkList;
void sortx(int x[],int num){
	int i=0,j=0,t;
	for(i=0;i<num-1;i++){
		for(j=i+1;j<num;j++){
			if(x[i]<x[j]){
			    t=x[i];x[i]=x[j];x[j]=t;}
		}
	}
}
struct node* indexFind(int x[],int sortx[],int num){   //C++中函数只能返回一个值，此处返回为指针（链表）
	int i;LinkList *head,*tail,*newnode;        //链表插入必须用三个结点，否则会产生地址冲突
	head=new node;
	tail=head;
	if(sortx[0]!=sortx[1]){
		for(i=0;i<num;i++){
			if(x[i]==sortx[0]){
				newnode=new node;newnode->data=i;tail->next=newnode;tail=newnode;break;}
		}
		tail->next=NULL;
	}
	else{
	    for(i=0;i<num;i++){
			if(x[i]==sortx[0]){
				newnode=new node;newnode->data=i;tail->next=newnode;tail=newnode;}
		}
		tail->next=NULL;
	}
	return head->next;   //表头为空链表
}
void get_cl(int t1[],int t2[],int cl[nt+2*nc]){
	int i,j,ii,nodeNum[2]={0};
	LinkList *head1,*head2;
	for(i=0;i<nt;i++){
		nodeNum[0]=i+1;nodeNum[1]=i+1;
		head1=indexFind(t1,nodeNum,nc);head2=indexFind(t2,nodeNum,nc);  //将nt+c*2+p中的c数存在链表中，而p对应0和1 (课件中与C语言下标对应，公式需全部减1)
		ii=i;
		while(head1!=NULL){
		    cl[ii]=nt+2*head1->data;ii=cl[ii];
			head1=head1->next;
		}
		while(head2!=NULL){
		    cl[ii]=nt+2*head2->data+1;ii=cl[ii];
			head2=head2->next;
		}
		cl[ii]=0;
	}
}
void get_c(int t1[],int t2[],int step[],int cl[],int cmax[],int csum[],int sortMax[],int sortSum[]){   //每个通信步完成后，再重新调用该函数更新csum和cmax
	int i,ii,is,nodeNum[2]={0},max1[nc]={0},max2[nc]={0};
	LinkList *head;
	for(ii=0;ii<nc;ii++){
		cmax[ii]=0;csum[ii]=0;sortMax[ii]=0;sortSum[ii]=0;}   //先进行归零操作
	head=indexFind(step,nodeNum,nc);
	while(head!=NULL){
		i=head->data;
		ii=t1[i]-1;
		while(cl[ii]!=0){
	        is=(cl[ii]-nt)/2;   //is即为节点对应通信下标，在每个通信步内设置禁区时，也就是对 step 的改变需要使用
			if(step[is]<=0){
			    max1[i]=max1[i]+1;}
			ii=cl[ii];
		}
		ii=t2[i]-1;
		while(cl[ii]!=0){
	        is=(cl[ii]-nt)/2;
			if(step[is]<=0){
			    max2[i]=max2[i]+1;}
			ii=cl[ii];
		}
		head=head->next;

		csum[i]=max1[i]+max2[i];
		if(max1[i]>max2[i]){
		    cmax[i]=max1[i];}
		else{
		    cmax[i]=max2[i];}
		sortMax[i]=cmax[i];sortSum[i]=csum[i];
	}
}
void ban_step(int t1[],int t2[],int cl[],int step[],int istep){
	int ii,iban;
	ii=t1[istep]-1;
	while(cl[ii]!=0){
		iban=(cl[ii]-nt)/2;ii=cl[ii];
		if(step[iban]==0){step[iban]= -1;}  //注意ban错前面已安排的通信
	}
	ii=t2[istep]-1;
	while(cl[ii]!=0){
		iban=(cl[ii]-nt)/2;ii=cl[ii];
		if(step[iban]==0){step[iban]= -1;}
	}
}
int step1(int cmax[],int csum[],int sortMax[],int sortSum[]){
	int i,istep,Sm=0;
	LinkList *cmax_head;
	sortx(sortMax,nc);sortx(sortSum,nc);
	cmax_head=indexFind(cmax,sortMax,nc);
	while(cmax_head!=NULL){
		i=cmax_head->data;
		if(Sm<csum[i]){
			Sm=csum[i];istep=i;}
		cmax_head=cmax_head->next;
	}
	return istep;
}
int step2(int w[],int csum[],int wistep,int step[]){
	int i,wSmin,istep,dw[nc],sortdw[nc];
	LinkList *dw_head;
	for(i=0;i<nc;i++){
		dw[i]= 1000000-abs(wistep-w[i]);sortdw[i]=dw[i];
		if(step[i]!=0){              //注意要除去原有已经安排上的通信的w比较
			dw[i]=0;sortdw[i]=0;}
	}
	sortx(sortdw,nc);
	dw_head=indexFind(dw,sortdw,nc);
	wSmin=100000000;
	while(dw_head!=NULL){
		i=dw_head->data;
		if(wSmin>csum[i]){
			istep=i;wSmin=csum[i];}
		dw_head=dw_head->next;
	}
	return istep;
}
bool isStep(int step[],int num){
	int i;
	for(i=0;i<num;i++){
		if(step[i]==0){
		    return true;}
	}
	return false;
}
int main(){    //计算总量、各节点计算量、平衡度up/down（up=sum(w)/sum(V)，down=max(Wi/Vi)）
	int i,k,istep,cl[nt+2*nc]={0};
	int step[nc]={0},cmax[nc],sortMax[nc],csum[nc],sortSum[nc],t1[nc]={1,2,3,1,2,2,3,4,5,6,5,6,8},t2[nc]={2,3,4,5,5,6,6,7,6,7,8,8,7};
	int w[nc]={10,10,10,5,15,5,20,20,9,9,10,20,20},wistep;
	
	get_cl(t1,t2,cl);   //(cl-nt)/2 取整即为对应通信的下标
	
	i=1;
	while(isStep(step,nc)){
		cout<<"the number of i="<<i<<endl;
		get_c(t1,t2,step,cl,cmax,csum,sortMax,sortSum);
		istep=step1(cmax,csum,sortMax,sortSum);  //分配准则1函数，前一步已通过step对cmax和csum处理，故此处不用
		cout<<"istep1 is:"<<istep<<endl;
		ban_step(t1,t2,cl,step,istep);
		step[istep]=i;   //第i个步骤里的第1个通信步找完，并标记完禁区

		wistep=0;k=1;
		while(isStep(step,nc)){
			wistep=(wistep+w[istep])/k;
			istep=step2(w,csum,wistep,step);  //分配准则2函数，注意需要对csum按照step进行相应下标禁区
			cout<<"istep2 is:"<<istep<<endl;
			ban_step(t1,t2,cl,step,istep);
			step[istep]=i;   //第i个步骤里的第2个通信步找完，并标记完禁区，需要重复该步骤
			k=k+1;
		}
		for(k=0;k<nc;k++){   //开放通信禁区
			if(step[k]==-1){
				step[k]=0;}
		}
		i=i+1;
	}
	for(k=0;k<nc;k++){
		cout<<step[k]<<'\t';}

	//最后根据step生成两个矩阵sc和st
	i=i-1;
	cout<<endl<<"The sum of step is:"<<i<<endl;

	int **sc=new int*[nt],**st=new int*[nt];
	for(k=0;k<nt;k++){
		sc[k]=new int[i];st[k]=new int[i];
	}



	system("pause");
	return 0;
}