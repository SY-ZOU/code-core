## Shell编程

- 简单命令

```shell
cat file1
```

- 多条命令

```shell
who; date; ps
```

- 复合命令

```shell
ps –e | grep student2
(ls ; cat file3 ; pwd) > run_log
```

- 后台命令

```shell
ls –lR /home/teacher > tlist &
```

## **构件原语**

### 1. **I/O**重定向

**一个进程通常(default)打开三个文件：**

- **标准输入文件（fd=0）**

- **标准输出文件（fd=1）**

- **标准错误输出文件（fd=2）**

```shell
grep abc 
grep abc < file1  #file1作为输入
grep abc < file1 > file2  #file1作为输入，file2作为输出
grep abc < file1 > file2 2> file3
```

### 2.**管道**

**A进程的输出作为B进程的输入**。------------> 管道作为存储A的输出，让B读取。

```shell
ps -e | grep student3 #查看当前系统中与用户student3相关的进程
```

