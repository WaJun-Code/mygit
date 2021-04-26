#include<iostream>
#include<iomanip>
using namespace std;

typedef struct node{
    int data;
    struct node* left;
    struct node* right;
}Node;    //与链表的结构定义类似 

class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        return ismirror(root,root);
    }
    bool ismirror(TreeNode* p,TreeNode* q){
        if(!p&&!q)
        {return true;}
        if(!p||!q)
        {return false;}
        if(p->data==q->data){
            return ismirror(p->left,q->right)&&ismirror(p->right,q->left);
        }
        else{
            return false;
        }
    }
};

void preorder(Node* node){
    if (node != NULL){
        printf("%d\n",node->data);
        preorder(node->left);
        preorder(node->right);}
    else{
        cout<<"It is an empty Node"<<endl;}
}

void main(){
    Node n1;Node n2;Node n3;Node n4;
    n1.data=5;n2.data=6;n3.data=7;n4.data=8;
    n1.left=&n2;n1.right=&n3;n2.left=&n4;n2.right=NULL;
    n3.left=NULL;n3.right=NULL;n4.left=NULL;n4.right=NULL;
    preorder(&n1);
	system("pause");
}