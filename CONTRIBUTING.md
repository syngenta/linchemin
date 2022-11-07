
# Secret management in development

# Architecture visualization   
pydeps requiring graphviz to be installed (not via pip!)
https://medium.com/illumination/visualize-dependencies-between-python-modules-d6e8e9a92c50
https://pythonawesome.com/python-module-dependency-visualization/

> pydeps src/linchemin  -o src/linchemin_dependency_diagram.svg   
> pydeps src/linchemin  -o src/linchemin_dependency_diagram.svg --cluster --max-cluster-size=3 --min-cluster-size=2 --keep-targe
t-cluster


# Keeping track of work 
## Use branching while work with featurs/bugs
In your Github fork, you need to keep your master branch clean, by clean I mean without any changes, like that you can create at any time a branch from your master. Each time that you want to commit a bug or a feature, you need to create a branch for it, which will be a copy of your master branch.

https://github.com/Kunena/Kunena-Forum/wiki/Create-a-new-branch-with-git-and-manage-branches  

## Reference issues in your development work

https://support.atlassian.com/jira-software-cloud/docs/reference-issues-in-your-development-work/  

###Branches  
Include the issue key at the beginning of the branch name when you create the branch to link it to your Jira issue.  
>git checkout -b JRA-123-branch-name

###Commits 
Include the issue key at the beginning of the commit message to link the commit to your Jira issue.    
>git commit -m "JRA-123 commit message"  

###Pull requests
Do at least one of the following:
Include a commit in the pull request that has the issue key in the commit message. Note, the commit cannot be a merge commit.
* Include the issue key at the beginning of the pull request title.
* Ensure that the source branch name also includes the issue key at the beginning of the branch name.
* This works by default in connected Bitbucket, GitLab, GitHub, and GitHub Enterprise tools.
If you create the pull request from the development panel in a Jira issue, the issue key is added automatically