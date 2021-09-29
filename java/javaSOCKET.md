## 网络操作IO

> InetAddress：用于表示网络上的硬件资源，即 IP 地址；
>
> URL：统一资源定位符，可通过IO读取写入数据；
>
> Sockets：使用 TCP 协议实现网络通信；
>
> Datagram：使用 UDP 协议实现网络通信;

### 1. InetAddress

- 没有公有的构造函数，只能通过静态方法来创建实例。
- `java.net.InetAddress`类是Java对IP地址（包括IPv4和IPv6）的高层表示。
- `InetAddress` 类提供将**主机名解析为其 IP 地址（或反之）的方法**。

### 2. URL

- 可以直接从 URL 中读取字节流数据。
- URL类中包含了很多方法用于访问URL的各个部分(协议，端口号，主机等等)

```java
try
{
  URL url = new URL("http://www.runoob.com/index.html?language=cn#j2se");
  System.out.println("URL 为：" + url.toString());
  System.out.println("协议为：" + url.getProtocol());
  System.out.println("验证信息：" + url.getAuthority());
  System.out.println("文件名及请求参数：" + url.getFile());
  System.out.println("主机名：" + url.getHost());
  System.out.println("路径：" + url.getPath());
  System.out.println("端口：" + url.getPort());
  System.out.println("默认端口：" + url.getDefaultPort());
  System.out.println("请求参数：" + url.getQuery());
  System.out.println("定位位置：" + url.getRef());
}catch(IOException e)
{
  e.printStackTrace();
}

URL 为：http://www.runoob.com/index.html?language=cn#j2se
协议为：http
验证信息：www.runoob.com
文件名及请求参数：/index.html?language=cn
主机名：www.runoob.com
路径：/index.html
端口：-1  //没有指明端口号，则返回-1
默认端口：80
请求参数：language=cn
定位位置：j2se
```

- **使用URL读取网络上的内容`url.openStream()`。**

```java
try{
  URL url = new URL("https://supermancantfly.top/markdown/");
  //获取url资源所表示的字节流
  InputStream in = url.openStream();
  //将字节流转换为字符流
  InputStreamReader reader = new InputStreamReader(in); //默认utf8编码，否则乱码
  //为字节流设置缓冲
  BufferedReader br = new BufferedReader(reader);
  String data;
  while((data=br.readLine())!=null){
    System.out.println(data);
  }
  br.close();reader.close();in.close();

} catch (MalformedURLException e) {
  e.printStackTrace();
} catch (IOException e) {
  e.printStackTrace();
}
```

### 3. URLConnections

- `openConnection() `返回一个 `java.net.URLConnection`。

- 如果你连接HTTP协议的`URL`, `openConnection()` 方法返回 `HttpURLConnection` 对象。
- 如果你连接的URL为一个 `JAR` 文件, `openConnection() `方法将返回` JarURLConnection `对象。

```java
//get请求

URL url = new URL(httpurl);
HttpURLConnection conn = (HttpURLConnection) url.openConnection();
//设置连接方式
conn.setRequestMethod("GET");
//设置主机连接时间超时时间3000毫秒
conn.setConnectTimeout(3000);
//设置读取远程返回数据的时间3000毫秒
conn.setReadTimeout(3000);
//发送请求
conn.connect();
//获取输入字节流
InputStream is = conn.getInputStream();
//封装输入字节流->字符流->缓冲流
BufferedReader br = new BufferedReader(new InputStreamReader(is, "UTF-8"));
//接收读取数据
StringBuffer sb = new StringBuffer();
String line = null;
while((line = br.readLine())!=null) {
  sb.append(line);
  sb.append("\r\n");
}
if(null!=br) {
  br.close();
}
if(null!=is) {
  is.close();
}
//关闭连接
conn.disconnect();
return sb.toString();

//post请求

String jsonparam = JSON.toString(param);
URL url = new URL(httpurl);
//获取httpurlConnection连接
HttpURLConnection conn = (HttpURLConnection) url.openConnection();
//设置读取超时
conn.setConnectTimeout(3000);
//设置读取超时
conn.setReadTimeout(3000);
//传送数据
conn.setDoOutput(true);
//读取数据
conn.setDoInput(true);
//设置请求方式
conn.setRequestMethod("POST");
//设置传入参数格式
conn.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
// 设置鉴权信息：Authorization: Bearer da3efcbf-0845-4fe3-8aba-ee040be542c0
conn.setRequestProperty("Authorization", "Bearer da3efcbf-0845-4fe3-8aba-ee040be542c0");
//获取输出流
OutputStream os = conn.getOutputStream();
//输出数据
os.write(jsonparam.getBytes());
//获取输入流
InputStream is = conn.getInputStream();
//封装输入流
BufferedReader br = new BufferedReader(new InputStreamReader(is,"UTF-8"));
StringBuffer sb = new StringBuffer();
String line = null;
while((line = br.readLine())!=null) {
  sb.append(line);
  sb.append("\r\n");
}
if(null != br) {
  br.close();
}
if(null != is) {
  is.close();
}
if(null != os) {
  os.close();
}
//关闭连接
conn.disconnect();
return sb.toString();
```

### 4. HttpClient（Apache库）

```java
//get

//获取httpclient对象
HttpClient client = HttpClientBuilder.create().build();
//获取get请求对象
HttpGet httpGet = new HttpGet(url);
HttpResponse response = null;
try {
  //3.执行get请求并返回结果
  response = httpclient.execute(httpget);
} catch (IOException e1) {
  e1.printStackTrace();
}
String result = null;
try {
  //4.处理结果，这里将结果返回为字符串
  HttpEntity entity = response.getEntity();
  if (entity != null) {
    result = EntityUtils.toString(entity,"utf-8");
  }
} catch (ParseException | IOException e) {
  e.printStackTrace();
} finally {
  try {
    response.close();
  } catch (IOException e) {
    e.printStackTrace();
  }
}
return result;

//post url/map

HttpClient client = HttpClientBuilder.create().build();
List<NameValuePair> formparams = new ArrayList<NameValuePair>();
for (Map.Entry<String, String> entry : map.entrySet()) {
  //给参数赋值
  formparams.add(new BasicNameValuePair(entry.getKey(), entry.getValue()));
}
UrlEncodedFormEntity entity = new UrlEncodedFormEntity(formparams, Consts.UTF_8);
HttpPost httppost = new HttpPost(url);
httppost.setEntity(entity);
HttpResponse response = null;
try {
  response = httpclient.execute(httppost);
} catch (IOException e) {
  e.printStackTrace();
}
HttpEntity entity1 = response.getEntity();
String result = null;
try {
  result = EntityUtils.toString(entity1);
} catch (ParseException | IOException e) {
  e.printStackTrace();
}
return result;
```

## SOCKET

> IP地址和端口号组成了所谓的SOCKET，是TCP与UDP基础。
>
> 套接字使用TCP提供了两台计算机之间的通信机制。 **客户端程序创建套接字，并尝试连接服务器的套接字。**
>
> Socket是进程通讯的一种方式，即调用一些API函数实现分布在不同主机的相关进程之间的数据交换。

### 1. SOCKET建立

- `java.net.Socket `类代表一个套接字，并且` java.net.ServerSocket` 类为服务器程序提供了一种来**监听客户端**，并与他们建立连接的机制。
- 连接建立后，通过使用 I/O 流在进行通信，**每一个socket都有一个输出流和一个输入流，客户端的输出流连接到服务器端的输入流，而客户端的输入流连接到服务器端的输出流**。

![](https://camo.githubusercontent.com/3371f62d744525df3ede0684ab932fe6aa52a188/68747470733a2f2f63732d6e6f7465732d313235363130393739362e636f732e61702d6775616e677a686f752e6d7971636c6f75642e636f6d2f31653661666663342d313865352d343539362d393665662d6662383463363362663838612e706e67)

### 2. TCP编程

- TCP 是一个**全双工**的通信协议，因此数据可以通过两个数据流在**同一时间**发送。以下是一些类提供的一套完整的有用的方法来实现 socket。

> **socket函数**：表示你买了或者借了一部手机。 
>
> **bind函数**：告诉别人你的手机号码，让他们给你打电话。 
>
> **listen函数**：打开手机的铃声，而不是静音，这样有电话时可以立马反应。listen函数的第二个参数，最大连接数，表示最多有几个人可以同时拨打你的号码。不过我们的手机，最多只能有一个人打进来，要不然就提示占线。 
>
> **connect函数**：你的朋友知道了你的号码，通过这个号码来联系你。在他等待你回应的时候，不能做其他事情，所以connect函数是阻塞的。 
>
> **accept函数**：你听到了电话铃声，接电话，accept it！然后“喂”一声，你的朋友听到你的回应，知道电话已经打进去了。至此，一个TCP连接建立了。 
>
> **read/write函数**：连接建立后，TCP的两端可以互相收发消息，这时候的连接是全双工的。对应打电话中的电话煲。 
>
> **close函数**：通话完毕，一方说“我挂了”，另一方回应"你挂吧"，然后将连接终止。实际的close(sockfd)有些不同，它不止是终止连接，还把手机也归还，不在占有这部手机，就当是公用电话吧。
>
> 连接是阻塞的，你一次只能响应一个用户的连接请求，怎么办？在你打电话到10086时，总服务台会让一个接线员来为你服务，而它自己却继续监听有没有新的电话接入。在网络编程中，这个过程类似于fork一个子进程，建立实际的通信连接，而主进程继续监听。 实际网络编程中，处理并发的方式还有select/poll/epoll等。

![](https://github.com/frank-lam/fullstack-tutorial/raw/master/notes/JavaArchitecture/assets/tcpsocket.png)

### 3. TCP实例

- 服务器监听

```java
int count = 0;//记录客户端数量
try {
  //1.创建一个serversocket并监听
  ServerSocket serverSocket = new ServerSocket(9099);
  System.out.println("服务器启动等待连接");
  Socket socket =null;
  //循环监听等待客户端连接,调用accept方法，等待连接，此时处于阻塞状态
  while(true){
    socket = serverSocket.accept();
    ServerThread serverThread = new ServerThread(socket);
    serverThread.run();
    count++;
    System.out.println("客户端数目为"+count);
  }

} catch (IOException e) {
  e.printStackTrace();
}
```

- 服务器socket线程
  - 设置线程优先级，否则会很慢。
  - 关闭了输出流则socket会关闭，一般不关闭输出流，直接关闭socket即可。

```java
/**
 * 服务器线程处理类
 */
public class ServerThread extends Thread {
		//socket资源
    Socket socket = null;
  
    public ServerThread(Socket socket){
        this.socket=socket;
    }

    public void run(){
        InputStream is = null;
        BufferedReader br = null;
        OutputStream os = null;
        PrintWriter pw = null;
        try {
            //获取输入流，读取客户端发送的信息
            is = socket.getInputStream();
            br = new BufferedReader(new InputStreamReader(is));
            StringBuffer info = new StringBuffer();
            String c;
            while((c=br.readLine())!=null){
                info.append(c);
                info.append("\n");
            }
            System.out.println("客户端说：\n"+info);
            //关闭输入流
            socket.shutdownInput();

            //获取输出流，响应客户端
            os = socket.getOutputStream();
            pw = new PrintWriter(os);
            pw.write("欢迎你");
            pw.flush();

            socket.close();socket.close();
        }catch (IOException e) {
            e.printStackTrace();
        }finally {
            try {
                //关闭资源
                if(is!=null)
                    is.close();
                if(br!=null)
                    br.close();
                if(pw!=null)
                    pw.close();
                if(os!=null)
                    os.close();
                if(socket!=null)
                    socket.close();
            }catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
```

- 客户端

```java
try {
  //指定服务器地址端口
  Socket socket = new Socket("localhost",9099);
  //获取输出流，向服务器发送信息
  OutputStream os = socket.getOutputStream();
  PrintWriter pw = new PrintWriter(os); //包装成打印流
  pw.println("用户名：12");
  pw.println("密码：12");
  pw.flush();//刷新
  //关闭输出流
  socket.shutdownOutput();
  //获取服务器输入流
  InputStream is = socket.getInputStream();
  BufferedReader br = new BufferedReader(new InputStreamReader(is));
  StringBuffer in = new StringBuffer();
  String c;
  while((c=br.readLine())!=null){
    in.append(c);
    in.append("\n");
  }
  System.out.println("我是客户端，服务器说：\n"+in);
  //关闭资源
  os.close();pw.close();is.close();br.close();
  socket.close();
} catch (UnknownHostException e) {
  e.printStackTrace();
} catch (IOException e) {
  e.printStackTrace();
}
```

### 4. UDP编程

- 服务器

```java
try {
  //创建DatagramSocket，指定端口
  DatagramSocket datagramSocket = new DatagramSocket(9999);
  //创建数据报，用于接受客户端数据
  byte[] data = new byte[1024];
  DatagramPacket datagramPacket = new DatagramPacket(data,data.length);
  //接受数据，在接受数据报之前一直阻塞
  datagramSocket.receive(datagramPacket);
  //读取数据
  String info = new String(data,0,datagramPacket.getLength());
  System.out.println("客户端说："+info);
  //响应客户端，获取客户端地址
  InetAddress address = datagramPacket.getAddress();
  int port = datagramPacket.getPort();
  byte[] data2 = "欢迎".getBytes("utf8");
  //创建数据报，响应发送
  DatagramPacket datagramPacket1 = new DatagramPacket(data2,data2.length,address,port);
  datagramSocket.send(datagramPacket1);
  //关闭socket
  datagramSocket.close();

}catch (IOException e){
  e.printStackTrace();
}
```

- 客户端

```java
try{
  //定义服务器地址，创建数据报
  byte[] data = "用户名：123；密码：1123".getBytes();
  InetAddress address = InetAddress.getByName("localhost");
  DatagramPacket datagramPacket = new DatagramPacket(data,data.length,address,9999);
  //创建socket
  DatagramSocket socket = new DatagramSocket();
  //发送
  socket.send(datagramPacket);
  //接受服务器返回数据
  byte[] data1 = new byte[1024];
  //返回数据保存在数据报
  DatagramPacket datagramPacket1 = new DatagramPacket(data1,data1.length); 
  socket.receive(datagramPacket1);
  String reply = new String(data1,0,datagramPacket1.getLength(),"utf8");
  System.out.println(reply);
  //关闭资源
  socket.close();
}catch (IOException e){
}
```

## IO模型

> 同步 I/O：将数据从内核缓冲区复制到应用进程缓冲区的阶段（第二阶段），应用进程会阻塞。使用同步IO时，Java自己处理IO读写。【亲自去取钱】
>
> 异步 I/O：第二阶段应用进程不会阻塞。使用异步IO时，Java将IO读写委托给OS处理，需要将数据缓冲区地址和大小传给OS，OS需要支持异步IO操作API。【委托小弟拿银行卡到银行取钱，然后给你】
>
> 阻塞I/O：应用进程被阻塞，直到数据从内核缓冲区复制到应用进程缓冲区中才返回。【 ATM排队取款，你只能等待】
>
> 非阻塞I/O：应用进程执行系统调用之后，内核返回一个错误码。应用进程可以继续执行，但是需要不断的执行系统调用来获知 I/O 是否完成，这种方式称为轮询（polling）。

- **一个输入操作通常包括两个阶段**
  - **等待数据准备好（等待数据从网络中到达套接字。到达时，它被复制到内核中的某个缓冲区【套接字】）**
  - **从内核向进程复制数据（从套接字读取数据流）**

- **Unix 有五种 I/O 模型：**
  - 阻塞式 I/O
  - 非阻塞式 I/O
  - I/O 复用（select 和 poll）
  - 信号驱动式 I/O（SIGIO）
  - 异步 I/O（AIO）

### 1. 阻塞式 I/O

- 在阻塞的过程中，其它应用进程还可以执行，因此阻塞不意味着整个操作系统都被阻塞。因为其它应用进程还可以执行，所以不消耗 CPU 时间，**这种模型的 CPU 利用率会比较高。**

```java
//用于接收 Socket 传来的数据，并复制到应用进程的缓冲区 buf 中。这里把 recvfrom() 当成系统调用。
ssize_t recvfrom(int sockfd, void *buf, size_t len, int flags, struct sockaddr *src_addr, socklen_t *addrlen);
```

![](https://camo.githubusercontent.com/5ebdb46341969caa39d2037f9061d966dbfd9961/68747470733a2f2f63732d6e6f7465732d313235363130393739362e636f732e61702d6775616e677a686f752e6d7971636c6f75642e636f6d2f313439323932383431363831325f342e706e67)

### 2. 非阻塞式 I/O

> 阻塞IO一个线程只能处理一个IO流事件，要想同时处理多个IO流事件要么多线程要么多进程
>
> 非阻塞IO可以一个线程处理多个流事件，只要不停地询所有流事件即可，当然这个方式也不好，当大多数流没数据时，也是会大量浪费CPU资源；
>
> 为了避免CPU空转，引进代理(select和poll，两种方式相差不大)，代理可以观察多个流I/O事件，空闲时会把当前线程阻塞掉，当有一个或多个I/O事件时，就从阻塞态醒过来，把所有IO流都轮询一遍，于是没有IO事件我们的程序就阻塞在select方法处，即便这样依然存在问题，我们从select出只是知道有IO事件发生，却不知道是哪几个流，还是只能轮询所有流，**epoll**这样的代理就可以把哪个流发生怎样的IO事件通知我们；　

- 由于 CPU 要处理更多的系统调用，因此这种模型的 CPU 利用率比较低。
- **NIO的特点是用户进程需要不断的主动询问kernel数据好了没有。**

![](https://camo.githubusercontent.com/d0fbceea06e5674972700d461ca62d6f1b715f0b/68747470733a2f2f63732d6e6f7465732d313235363130393739362e636f732e61702d6775616e677a686f752e6d7971636c6f75642e636f6d2f313439323932393030303336315f352e706e67)

### 3. I/O 复用(阻塞)

- **使用 select 或者 poll 等待数据，并且可以等待多个套接字中的任何一个变为可读。这一过程会被阻塞，当某一个套接字可读时返回，之后再使用 recvfrom 把数据从内核复制到进程中。**

- 每次select阻塞结束返回后，可以获得多个准备就绪的套接字（即一个select可以对多个套接字进行管理，类似于同时监控多个套接字事件是否就绪）
- 如果一个 Web 服务器没有 I/O 复用，那么每一个 Socket 连接都需要创建一个线程去处理。如果同时有几万个连接，那么就需要创建相同数量的线程。**相比于多进程和多线程技术，I/O 复用不需要进程线程创建和切换的开销，系统开销更小。**
- 会阻塞进程，但和IO阻塞不同，这些函数可以同时阻塞多个IO操作，而且可以同时对多个读操作，写操作IO进行检验，直到有数据到达，才真正调用IO操作函数。
- IO多路复用的优势在于并发数比较高的IO操作情况，可以同时处理多个连接，和bloking IO一样socket是被阻塞的，只不过在多路复用中socket是被select阻塞，而在阻塞IO中是被socket IO给阻塞。

![](https://camo.githubusercontent.com/c31f8db408e14826915b8d7b70724e5095298ee0/68747470733a2f2f63732d6e6f7465732d313235363130393739362e636f732e61702d6775616e677a686f752e6d7971636c6f75642e636f6d2f313439323932393434343831385f362e706e67)

### 4. 信号驱动 I/O(非阻塞)

- 应用进程使用 sigaction 系统调用，内核立即返回，应用进程可以继续执行，也就是说等待数据阶段应用进程是非阻塞的。内核在数据到达时向应用进程发送 SIGIO 信号，**应用进程收到之后在信号处理程序中调用 recvfrom 将数据从内核复制到应用进程中。**
- **相比于非阻塞式 I/O 的轮询方式，信号驱动 I/O 的 CPU 利用率更高。**

![](https://camo.githubusercontent.com/9533dfd9ce5b31d63b70ba6ce1aeae1ae64958db/68747470733a2f2f63732d6e6f7465732d313235363130393739362e636f732e61702d6775616e677a686f752e6d7971636c6f75642e636f6d2f313439323932393535333635315f372e706e67)

### 5. 异步 I/O（非阻塞）

- **应用进程执行 aio_read 系统调用会立即返回，应用进程可以继续执行，不会被阻塞，内核会在所有操作完成之后向应用进程发送信号。**

- 异步 I/O 与信号驱动 I/O 的区别在于，异步 I/O 的信号是通知应用进程 I/O 完成，而信号驱动 I/O 的信号是通知应用进程可以开始 I/O。
- 同步 I/O：将数据从内核缓冲区复制到应用进程缓冲区的阶段（第二阶段），应用进程会阻塞。（上述四个）
- 异步 I/O：第二阶段应用进程不会阻塞。

![](https://camo.githubusercontent.com/9c1afa0a4d217e0adfc91ab3b4d7ea9f3d6b1463/68747470733a2f2f63732d6e6f7465732d313235363130393739362e636f732e61702d6775616e677a686f752e6d7971636c6f75642e636f6d2f313439323933303234333238365f382e706e67)

### 6. 对比

![](https://camo.githubusercontent.com/d89aed2ba6c5390aad0626b013c288d8849c4f39/68747470733a2f2f63732d6e6f7465732d313235363130393739362e636f732e61702d6775616e677a686f752e6d7971636c6f75642e636f6d2f313439323932383130353739315f332e706e67)

## IO复用

> I/O多路复用（multiplexing）的本质是通过一种机制（系统内核缓冲I/O数据），让单个进程可以监视多个文件描述符，一旦某个描述符就绪（一般是读就绪或写就绪），能够通知程序进行相应的读写操作。
>
> 文件句柄（文件描述符），其实就是一个整数，我们最熟悉的句柄是0、1、2三个，0是标准输入，1是标准输出，2是标准错误输出。0、1、2是整数表示的，对应的FILE *结构的表示就是stdin、stdout、stderr。POSIX标准规定，每次打开的文件时(含socket)必须使用当前进程中最小可用的文件描述符号码。

### 1. SELECT

```java
int select(int maxfdp1, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout);
FD_CLR(inr fd,fd_set* set)；//用来清除描述词组set中相关fd 的位
FD_ISSET(int fd,fd_set *set)；//用来测试描述词组set中相关fd 的位是否为真
FD_SET（int fd,fd_set*set）；//用来设置描述词组set中相关fd的位
FD_ZERO（fd_set *set）；//用来清除描述词组set的全部位
/**
 （1）int maxfdp1 指定待测试的文件描述符个数，它的值是待测试的最大文件描述符加1，因为select会无差别遍历整个文件描述符表直到找到目标，而文件描述符是从0开始的，所以一共是集合中的最大的文件描述符+1次。
 （2）fd_set *readset , fd_set *writeset , fd_set *exceptset。fd_set可以理解为一个集合，这个集合中存放的是文件描述符(file descriptor)，即文件句柄。
 （3）timeout 为超时参数，调用 select 会一直阻塞直到有描述符的事件到达或者等待的时间超过 timeout。
 （4）成功调用返回结果大于 0数目，出错返回结果为 -1，超时返回结果为 0。
**/
```

> 理解fd_set，取fd_set长度为1字节，fd_set中的每一bit可以对应一个文件描述符fd。则1字节长的fd_set最大可以对应8个fd。
>
> （1）执行fd_set set; FD_ZERO(&set);则set用位表示是0000,0000。
>
> （2）若fd＝5,执行FD_SET(fd,&set);后set变为0001,0000(第5位置为1)
>
> （3）若再加入fd＝2，fd=1,则set变为0001,0011
>
> （4）执行select(6,&set,null,null,time)阻塞等待
>
> （5）若fd=1,fd=2上都发生可读事件，则select返回，此时set变为0000,0011。注意：没有事件发生的fd=5被清空。

- **select 允许应用程序监视一组文件描述符，等待一个或者多个描述符成为就绪状态，从而完成 I/O 操作。**

- 由于select采用轮询的方式扫描文件描述符，文件描述符数量越多，性能越差.。

- **select返回的是含有整个句柄的数组，应用程序需要遍历整个数组才能发现哪些句柄发生了事件；**我们只能无差别轮询所有流，找出能读出数据，或者写入数据的流，对他们进行操作。所以**select具有O(n)的无差别轮询复杂度**，同时处理的流越多，无差别轮询时间就越长。

- **select需要复制大量的句柄数据结构，产生巨大的开销；**

  ```c
  int main()
  {
    /***********服务器的listenfd已经准本好了**************/
    fd_set readfds; // 读集合
    fd_set writefds; //写集合
    FD_ZERO(&readfds);
    FD_ZERO(&writefds);
    FD_SET(listenfd, &readfds); //监听套接字写入读集合
    fd_set temprfds = readfds;
    fd_set tempwfds = writefds;
    int maxfd = listenfd; //最大套接字，即个数-1（从0开始）
    int nready;
    char buf[MAXNFD][BUFSIZE] = {0}; //缓冲区 MAXFD最多文件套接字个数。 BUFSIZE缓冲区大小
    while(1){
      temprfds = readfds;
      tempwfds = writefds;
      nready = select(maxfd+1, &temprfds, &tempwfds, NULL, NULL);//阻塞读取
      if(FD_ISSET(listenfd, &temprfds))｛     
        //如果监听到的是listenfd就进行accept
        int sockfd = accept(listenfd, (struct sockaddr*)&clientaddr, &len);
        //将新accept的scokfd加入监听集合，并保持maxfd为最大fd
        FD_SET(sockfd, &readfds);
        maxfd = maxfd>sockfd?maxfd:sockfd;
        //如果意见检查了nready个fd，就没有必要再等了，直接下一个循环
        if(--nready==0)
          continue;
      }
       
      int fd = 0;
      //遍历文件描述符表，处理接收到的消息
      for(;fd<=maxfd; fd++)｛  
        if(fd == listenfd)
          continue;
        if(FD_ISSET(fd, &temprfds)){
          int ret = read(fd, buf[fd], sizeof buf[0]);
          if(0 == ret)｛  //客户端链接已经断开
            close(fd);
            FD_CLR(fd, &readfds);
            if(maxfd==fd) 
              --maxfd;
            continue;
          }
          //将fd加入监听可写的集合
          FD_SET(fd, &writefds); 
        }
        //找到了接收消息的socket的fd，接下来将其加入到监视写的fd_set中
        //将在下一次while()循环开始监视
        if(FD_ISSET(fd, &tempwfds)){
          int ret = write(fd, buf[fd], sizeof buf[0]);
          printf("ret %d: %d\n", fd, ret);
          FD_CLR(fd, &writefds);
        }
      }
    }
    close(listenfd);
  }
  ```

### 2. POLL

```c
int poll(struct pollfd *fds, unsigned int nfds, int timeout);
#poll 的功能与 select 类似，也是等待一组描述符中的一个成为就绪状态。
#poll 中的描述符是 pollfd 类型的数组，pollfd 的定义如下：
struct pollfd {
  int   fd;         /* file descriptor */
  short events;     /* requested events */
  short revents;    /* returned events */
};
```

- 相比select模型，poll使用链表保存文件描述符，因此没有了监视文件数量的限制。其余和SELECT一样。

### 3. EPOLL

> 在select/poll时代，服务器进程每次都把这100万个连接告诉操作系统(从用户态复制句柄数据结构到内核态)，让操作系统内核去查询这些套接字上是否有事件发生，轮询完后，再将句柄数据复制到用户态，让服务器应用程序轮询处理已发生的网络事件，这一过程资源消耗较大。
>
> epoll通过在Linux内核中申请一个简易的文件系统(文件系统一般用什么数据结构实现？B+树)。把原先的select/poll调用分成了3个部分：
>
> 1）调用epoll_create()建立一个epoll对象(在epoll文件系统中为这个句柄对象分配资源)
>
> 2）调用epoll_ctl向epoll对象中添加这100万个连接的套接字
>
> 3）调用epoll_wait收集发生的事件的连接
>
> 只需要在进程启动时建立一个epoll对象，然后在需要的时候向这个epoll对象中添加或者删除连接。同时，epoll_wait的效率也非常高，因为调用epoll_wait时，并没有一股脑的向操作系统复制这100万个连接的句柄数据，内核也不需要去遍历全部的连接。

```c
int epoll_create(int size);
//创建了红黑树和就绪链表；
int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event)；
//如果增加socket句柄，则检查在红黑树中是否存在，存在立即返回，不存在则添加到树干上，然后向内核注册回调函数，用于当中断事件来临时向准备就绪链表中插入数据；
int epoll_wait(int epfd, struct epoll_event * events, int maxevents, int timeout);
//立刻返回准备就绪链表里的数据即可。

//返回的epoll结构
struct eventpoll {
　　...
　　/*红黑树的根节点，这棵树中存储着所有添加到epoll中的事件，也就是这个epoll监控的事件*/
　　struct rb_root rbr;
　　/*双向链表rdllist保存着将要通过epoll_wait返回给用户的、满足条件的事件*/
　　struct list_head rdllist;
　　...
};
//epoll_event准备事件结构体和事件结构体数组
struct epoll_event   
  event.events
  event.data.fd 
```

- epoll_ctl() 用于向内核注册新的描述符或者是改变某个文件描述符的状态。已注册的描述符在内核**中会被维护在一棵红黑树上，通过回调函数内核会将 I/O 准备好的描述符加入到一个rdllist双向链表管理**，**当epoll_wait调用时，仅仅观察这个rdllist双向链表里有没有数据即可。有数据就返回**，没有数据就sleep，等到timeout时间到后即使链表没数据也返回。所以，epoll_wait非常高效。**（复杂度降低到了O(1)）**
- 添加到epoll中的事件都会与设备(如网卡)驱动程序**建立回调关系**，也就是说相应事件的发生时会调用这里的回调方法。这个回调方法在内核中叫做**ep_poll_callback**，它会把这样的事件放到上面的rdllist双向链表中。

- epoll_ctl在向epoll对象中添加、修改、删除事件时，**从rbr红黑树中查找事件也非常快，也就是说epoll是非常高效的，它可以轻易地处理百万级别的并发连接。**

- IO效率不随FD数目增加而线性下降，它只会对"活跃"的socket进行操作— 这是因为在内核实现中epoll是根据每个fd上面的callback函数实现的。那么，只有"活跃"的socket才会主动的去调用 callback函数，其他idle状态socket则不会，在这点上，**epoll实现了一个"伪"AIO，因为这时候推动力在os内核。**

```java
int main()
{
  /* ... */
  /* 创建epoll对象 */
  int epoll_fd = epoll_create(1024);
   
  //准备一个事件结构体
  struct epoll_event event = {0};
  event.events = EPOLLIN;
  event.data.fd = listenfd;  //data是一个共用体，除了fd还可以返回其他数据
   
  //ctl是监控listenfd是否有event被触发
  //如果发生了就把event通过wait带出。
  //所以，如果event里不标明fd，我们将来获取就不知道哪个fd
  epoll_ctl(epoll_fd, EPOLL_CTL_ADD, listenfd, &event);
   
  struct epoll_event revents[MAXNFD] = {0};
  int nready;
  char buf[MAXNFD][BUFSIZE] = {0};
  while(1){
    //wait返回等待的event发生的数目
    //并把相应的event放到event类型的数组中
    nready = epoll_wait(epoll_fd, revents, MAXNFD, -1)
    int i = 0;
    for(;i<nready; i++){
      //wait通过在events中设置相应的位来表示相应事件的发生
      //如果输入可用，那么下面的这个结果应该为真
      if(revents[i].events & EPOLLIN){
        //如果是listenfd有数据输入
        if(revents[i].data.fd == listenfd){
          int sockfd = accept(listenfd, (struct sockaddr*)&clientaddr, &len);
          struct epoll_event event = {0};
          event.events = EPOLLIN;
          event.data.fd = sockfd;
          epoll_ctl(epoll_fd, EPOLL_CTL_ADD, sockfd, &event);
        }
        else{
          int ret = read(revents[i].data.fd, buf[revents[i].data.fd], sizeof buf[0]);
          if(0 == ret){
            close(revents[i].data.fd);
            epoll_ctl(epoll_fd, EPOLL_CTL_DEL, revents[i].data.fd, &revents[i]);
          }
           
          revents[i].events = EPOLLOUT;
          epoll_ctl(epoll_fd, EPOLL_CTL_MOD, revents[i].data.fd, &revents[i]);
        }
      }
      else if(revents[i].events & EPOLLOUT){
        int ret = write(revents[i].data.fd, buf[revents[i].data.fd], sizeof buf[0]);
        revents[i].events = EPOLLIN;
        epoll_ctl(epoll_fd, EPOLL_CTL_MOD, revents[i].data.fd, &revents[i]);
      }
    }
  }
  close(listenfd);
}
```

### 4. EPOLL两种工作模式

-  LT 模式（默认）：当 epoll_wait() 检测到描述符事件到达时，将此事件通知进程，进程可以不立即处理该事件，下次调用 epoll_wait() 会再次通知进程。是默认的一种模式，并且同时支持 Blocking 和 No-Blocking。

- ET 模式：通知之后进程必须立即处理事件，下次再调用 epoll_wait() 时不会再得到事件到达的通知。**ET很大程度上减少了 epoll 事件被重复触发的次数，因此效率要比 LT 模式高。系统不会充斥大量你不关心的就绪文件描述符。**

![](https://img-blog.csdnimg.cn/2018110814591325.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RhYWlrdWFpY2h1YW4=,size_16,color_FFFFFF,t_70)

### 5. 区别

- select，poll实现需要自己不断轮询所有fd集合，直到设备就绪。而epoll其实也需要调用epoll_wait不断轮询就绪链表，**但是select和poll在“醒着”的时候要遍历整个fd集合，而epoll在“醒着”的时候只要判断一下就绪链表是否为空就行了，这节省了大量的CPU时间。这就是回调机制带来的性能提升。**
- select，poll每次调用都要把fd集合从用户态往内核态拷贝一次，而epoll只要一次拷贝。

### 6. 应用场景

- select 的 timeout 参数精度为微秒，而 poll 和 epoll 为毫秒，**因此 select 更加适用于实时性要求比较高的场景，比如核反应堆的控制。select 可移植性更好，几乎被所有主流平台所支持**。
- poll 没有最大描述符数量的限制，如果**平台支持并且对实时性要求不高，应该使用 poll 而不是 select。**
- Epoll只需要运行在 Linux 平台上，有大量的描述符需要同时轮询，并且这些连接最好是长连接。需要同时监控小于 1000 个描述符，就没有必要使用 epoll，因为这个应用场景下并不能体现 epoll 的优势。需要监控的描述符状态变化多，而且都是非常短暂的，也没有必要使用 epoll。因为 epoll 中的所有描述符都存储在内核中，造成每次需要对描述符的状态改变都需要通过 epoll_ctl() 进行系统调用，频繁系统调用降低效率。并且 epoll 的描述符存储在内核，不容易调试。



## BIO/NIO/AIO

### 1. BIO

- **BIO（同步阻塞）**：用户进程发起一个IO操作以后，必须等待IO操作的真正完成后，才能继续运行；客户端和服务器连接需要三次握手，使用简单，但吞吐量小。
- 服务器实现模式为一个连接一个线程，即客户端有连接请求时服务器端就需要启动一个线程进行处理，**如果这个连接不做任何事情会造成不必要的线程开销，当然可以通过线程池机制改善。**
- BIO方式适用于连接数目比较小且固定的架构，这种方式对服务器资源要求比较高，并发局限于应用中，JDK1.4以前的唯一选择，但程序直观简单易理解。
- BIO模型中通过**Socket**和**ServerSocket**完成套接字通道实现。阻塞，同步，连接耗时。
- 我们可以使用线程池来管理这些线程，实现1个或多个线程处理N个客户端的模型（但是底层还是使用的同步阻塞I/O），通常被称为“**伪异步I/O模型**“。

> 使用 CachedThreadPool 线程池（不限制线程数量），其实除了能自动帮我们管理线程（复用），看起来也就像是1:1的客户端：线程数模型，而使用 FixedThreadPool 我们就有效的控制了线程的最大数量，保证了系统有限的资源的控制，实现了N:M的伪异步 I/O 模型。
>
> 正因为限制了线程数量，如果发生大量并发请求，超过最大数量的线程就只能等待，直到线程池中的有空闲的线程可以被复用。而对 Socket 的输入流就行读取时，会一直阻塞，直到发生：
>
> - 有数据可读
> - 可用数据以及读取完毕
> - 发生空指针或 I/O 异常
>
> 所以在读取数据较慢时（比如数据量大、网络传输慢等），大量并发的情况下，其他接入的消息，只能一直等待，这就是最大的弊端。而后面即将介绍的NIO，就能解决这个难题。

![](https://github.com/frank-lam/fullstack-tutorial/raw/master/notes/JavaArchitecture/assets/java-bio-threadpool.png)

### 2. NIO

- **NIO（同步非阻塞）**：**用户进程发起一个IO操作以后，可做其它事情，但用户进程需要经常询问IO操作是否完成，这样造成不必要的CPU资源浪费；**客户端与服务器通过Channel连接，采用多路复用器轮询注册的Channel。提高吞吐量和可靠性。

- **服务器实现模式为一个请求一个线程，即客户端发送的连接请求都会注册到多路复用器上，多路复用器轮询到连接有I/O请求时才启动一个线程进行处理。**

- NIO方式适用于连接数目多且连接比较短（轻操作）的架构，比如聊天服务器，并发局限于应用中，编程比较复杂，JDK1.4开始支持。

- 流与块：I/O 与 NIO 最重要的区别是数据打包和传输的方式，I/O 以流的方式处理数据，而 NIO 以块的方式处理数据。

  - 面向流的 I/O 一次处理一个字节数据：面向流的 I/O 通常相当慢。
  - 面向块的 I/O 一次处理一个数据块，按块处理数据比按流处理数据要快得多。但是面向块的 I/O 缺少一些面向流的 I/O 所具有的优雅性和简单性。

```java
//文件读写例子
public static void fastCopy(String src, String dist) throws IOException {

    /* 获得源文件的输入字节流 */
    FileInputStream fin = new FileInputStream(src);

    /* 获取输入字节流的文件通道 */
    FileChannel fcin = fin.getChannel();

    /* 获取目标文件的输出字节流 */
    FileOutputStream fout = new FileOutputStream(dist);

    /* 获取输出字节流的文件通道 */
    FileChannel fcout = fout.getChannel();

    /* 为缓冲区分配 1024 个字节 */
    ByteBuffer buffer = ByteBuffer.allocateDirect(1024);

    while (true) {

        /* 从输入通道中读取数据到缓冲区中 */
        int r = fcin.read(buffer);

        /* read() 返回 -1 表示 EOF */
        if (r == -1) {
            break;
        }

        /* 切换读写 */
        buffer.flip();

        /* 把缓冲区的内容写入输出文件中 */
        fcout.write(buffer);

        /* 清空缓冲区 */
        buffer.clear();
    }
}
```

> **缓冲区Buffer**：BIO是将数据直接写入或读取到Stream对象中。而发送给一个通道的所有数据都必须首先放到缓冲区中，同样地，从通道中读取的任何数据都要先读到缓冲区中。Buffer最常见的类型是ByteBuffer，另外还有CharBuffer，ShortBuffer，IntBuffer，LongBuffer，FloatBuffer，DoubleBuffer。
>
> **通道Channel**：和流不同，通道是双向的。NIO可以通过Channel进行数据的读，写和同时读写操作。通道分为两大类：一类是网络读写（SelectableChannel），一类是用于文件操作（FileChannel），我们使用的SocketChannel和ServerSocketChannel都是SelectableChannel的子类。
>
> **多路复用器Selector**：NIO编程的基础。Selector会不断地轮询注册在其上的通道（Channel），如果某个通道处于就绪状态，会被Selector轮询出来，然后通过SelectionKey可以取得就绪的Channel集合，从而进行后续的IO操作。服务器端只要提供一个线程负责Selector的轮询，就可以接入成千上万个客户端。

![](https://github.com/frank-lam/fullstack-tutorial/raw/master/notes/JavaArchitecture/assets/java-nio.png)

```java
//套接字NIO实例
public class NIOServer {

    public static void main(String[] args) throws IOException {

        Selector selector = Selector.open();

        ServerSocketChannel ssChannel = ServerSocketChannel.open();
        ssChannel.configureBlocking(false);
        ssChannel.register(selector, SelectionKey.OP_ACCEPT);

        ServerSocket serverSocket = ssChannel.socket();
        InetSocketAddress address = new InetSocketAddress("127.0.0.1", 8888);
        serverSocket.bind(address);

        while (true) {

            selector.select();
            Set<SelectionKey> keys = selector.selectedKeys();
            Iterator<SelectionKey> keyIterator = keys.iterator();

            while (keyIterator.hasNext()) {

                SelectionKey key = keyIterator.next();

                if (key.isAcceptable()) {

                    ServerSocketChannel ssChannel1 = (ServerSocketChannel) key.channel();

                    // 服务器会为每个新连接创建一个 SocketChannel
                    SocketChannel sChannel = ssChannel1.accept();
                    sChannel.configureBlocking(false);

                    // 这个新连接主要用于从客户端读取数据
                    sChannel.register(selector, SelectionKey.OP_READ);

                } else if (key.isReadable()) {

                    SocketChannel sChannel = (SocketChannel) key.channel();
                    System.out.println(readDataFromSocketChannel(sChannel));
                    sChannel.close();
                }

                keyIterator.remove();
            }
        }
    }

    private static String readDataFromSocketChannel(SocketChannel sChannel) throws IOException {

        ByteBuffer buffer = ByteBuffer.allocate(1024);
        StringBuilder data = new StringBuilder();

        while (true) {

            buffer.clear();
            int n = sChannel.read(buffer);
            if (n == -1) {
                break;
            }
            buffer.flip();
            int limit = buffer.limit();
            char[] dst = new char[limit];
            for (int i = 0; i < limit; i++) {
                dst[i] = (char) buffer.get(i);
            }
            data.append(dst);
            buffer.clear();
        }
        return data.toString();
    }
}
public class NIOClient {

    public static void main(String[] args) throws IOException {
        Socket socket = new Socket("127.0.0.1", 8888);
        OutputStream out = socket.getOutputStream();
        String s = "hello world";
        out.write(s.getBytes());
        out.close();
    }
}
```

### 3.AIO

- **AIO（异步非阻塞）**：NIO的升级版，采用异步通道实现异步通信，其read和write方法均是异步方法。用户进程发起一个IO操作然后，立即返回，等IO操作真正的完成以后，应用程序会得到IO操作完成的通知。类比Future模式。
- 服务器实现模式为一个有效请求一个线程，客户端的I/O请求都是由OS先完成了再通知服务器应用去启动线程进行处理。
- AIO方式使用于连接数目多且连接比较长（重操作）的架构，比如相册服务器，充分调用OS参与并发操作，编程比较复杂，JDK7开始支持。

- AIO模型中通过**AsynchronousSocketChannel**和**AsynchronousServerSocketChannel**完成套接字通道实现。非阻塞，异步。



