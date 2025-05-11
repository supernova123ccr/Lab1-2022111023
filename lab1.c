#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <random>
#include <queue>
#include <limits>
#include <cmath>
#include <chrono>
#include <thread>
#include <unordered_set>
#include <utility>
#include <iomanip>

using namespace std;

using Graph = unordered_map<string, unordered_map<string, int>>;

// 清洗文本并返回清理后的文本和单词列表
pair<string, vector<string>> clean_text(const string &text)
{
    string cleaned;
    for (char c : text)
    {
        if (isalpha(c))
            cleaned += tolower(c);
        else
            cleaned += ' ';
    }

    stringstream ss(cleaned);
    vector<string> words;
    string word;

    while (ss >> word)
        words.push_back(word);

    return {cleaned, words};
}

// 构建图
Graph build_graph(const vector<string> &words)
{
    Graph graph;
    for (size_t i = 0; i < words.size() - 1; ++i)
    {
        string current = words[i];
        string next = words[i + 1];
        graph[current][next]++;
    }
    return graph;
}

// 显示有向图
void showDirectedGraph(const Graph &graph, const string &filename = "")
{
    // 文本显示
    for (const auto &[src, edges] : graph)
    {
        cout << src << " -> ";
        for (const auto &[dest, weight] : edges)
            cout << dest << "(" << weight << ") ";
        cout << endl;
    }

    // 生成DOT文件
    if (!filename.empty())
    {
        ofstream dot_file(filename + ".dot");
        dot_file << "digraph G {\n";
        for (const auto &[src, edges] : graph)
        {
            for (const auto &[dest, weight] : edges)
                dot_file << "  \"" << src << "\" -> \"" << dest
                         << "\" [label=\"" << weight << "\"];\n";
        }
        dot_file << "}\n";
        dot_file.close();
        system(("dot -Tpng " + filename + ".dot -o " + filename + ".png").c_str());
    }
}

// 查询桥接词
vector<string> queryBridgeWords(const Graph &g, string w1, string w2)
{
    vector<string> bridges;

    // 转换输入单词为小写
    transform(w1.begin(), w1.end(), w1.begin(), ::tolower);
    transform(w2.begin(), w2.end(), w2.begin(), ::tolower);

    if (!g.count(w1))
    {
        cout << "Warning: '" << w1 << "' not found in graph." << endl;
        return bridges;
    }
    if (!g.count(w2))
    {
        cout << "Warning: '" << w2 << "' not found in graph." << endl;
        return bridges;
    }

    const auto &neighbors = g.at(w1);
    for (const auto &[neighbor, _] : neighbors)
    {
        if (g.at(neighbor).count(w2))
        {
            bridges.push_back(neighbor);
        }
    }

    return bridges;
}

// 生成新文本（修正后的版本）
string generateNewText(const string &inputText, const Graph &g)
{
    auto [_, words] = clean_text(inputText); // 使用用户输入的新文本
    if (words.size() < 2)
        return inputText;

    random_device rd;
    mt19937 gen(rd());
    vector<string> result;

    for (size_t i = 0; i < words.size() - 1; ++i)
    {
        string word1 = words[i];
        string word2 = words[i + 1];
        result.push_back(word1);
        vector<string> bridges = queryBridgeWords(g, word1, word2);
        if (!bridges.empty())
        {
            uniform_int_distribution<> dis(0, bridges.size() - 1);
            result.push_back(bridges[dis(gen)]);
        }
    }
    result.push_back(words.back());

    string output;
    for (const auto &w : result)
    {
        output += w + " ";
    }
    if (!output.empty())
    {
        output.pop_back(); // 移除末尾空格
    }
    return output;
}

// 最短路径（Dijkstra算法）
pair<vector<string>, double> shortestPath(const Graph &g, const string &start, const string &end)
{
    unordered_map<string, double> dist;
    unordered_map<string, string> prev;
    using Pair = pair<double, string>;
    priority_queue<Pair, vector<Pair>, greater<>> pq;

    for (const auto &[node, _] : g)
        dist[node] = numeric_limits<double>::infinity();

    if (!g.count(start))
        return {{}, numeric_limits<double>::infinity()};

    dist[start] = 0;
    pq.emplace(0, start);

    while (!pq.empty())
    {
        auto [d, u] = pq.top();
        pq.pop();
        if (u == end)
            break;
        if (d > dist[u])
            continue;

        for (const auto &[v, w] : g.at(u))
        {
            if (dist[v] > dist[u] + w)
            {
                dist[v] = dist[u] + w;
                prev[v] = u;
                pq.emplace(dist[v], v);
            }
        }
    }

    if (dist[end] == numeric_limits<double>::infinity())
        return {{}, numeric_limits<double>::infinity()};

    vector<string> path;
    for (string at = end; !at.empty(); at = prev.count(at) ? prev[at] : "")
        path.push_back(at);

    reverse(path.begin(), path.end());
    return {path, dist[end]};
}

// PageRank计算
// 计算TF-IDF权重
unordered_map<string, double> compute_tfidf_weights(const Graph &g)
{
    unordered_map<string, double> weights;
    unordered_map<string, int> document_freq;

    // 统计每个词出现的文档数
    for (const auto &[src, edges] : g)
    {
        unordered_set<string> unique_words;
        for (const auto &[dest, _] : edges)
            unique_words.insert(dest);
        for (const auto &word : unique_words)
            document_freq[word]++;
    }

    // 计算TF-IDF
    const double N = g.size();
    for (const auto &[src, edges] : g)
    {
        const double total_edges = edges.size();
        double tfidf_sum = 0.0;
        unordered_map<string, double> tfidf_values;

        // 计算TF
        for (const auto &[dest, count] : edges)
        {
            double tf = count / total_edges;
            double idf = log(N / (1 + document_freq[dest]));
            tfidf_values[dest] = tf * idf;
            tfidf_sum += tf * idf;
        }

        // 归一化处理
        for (auto &[word, val] : tfidf_values)
            weights[src] += val / tfidf_sum;
    }

    return weights;
}

// 改进后的PageRank计算
unordered_map<string, double> pagerank(
    const Graph &g,
    double d = 0.85,
    int iter = 100,
    bool use_tfidf = false)
{
    unordered_map<string, double> pr;
    unordered_map<string, vector<string>> in_edges;
    vector<string> nodes;

    // 初始化节点和入边
    for (const auto &[u, edges] : g)
    {
        nodes.push_back(u);
        for (const auto &[v, _] : edges)
            in_edges[v].push_back(u);
    }

    // 初始化PR值
    if (use_tfidf)
    {
        auto initial_weights = compute_tfidf_weights(g);
        double sum = 0.0;
        for (const auto &[k, v] : initial_weights)
            sum += v;
        for (auto &[k, v] : initial_weights)
            v /= sum; // 归一化
        pr = initial_weights;
    }
    else
    {
        double init = 1.0 / nodes.size();
        for (const auto &n : nodes)
            pr[n] = init;
    }

    for (int i = 0; i < iter; ++i)
    {
        unordered_map<string, double> new_pr;
        double dangling = 0.0;

        // 计算悬挂节点贡献
        for (const auto &n : nodes)
            if (g.at(n).empty())
                dangling += pr[n];

        const double teleport = (1 - d) / nodes.size();
        const double dangling_contribution = d * dangling / nodes.size();

        for (const auto &n : nodes)
        {
            double sum = 0.0;
            for (const auto &u : in_edges[n])
                sum += pr[u] / g.at(u).size();

            new_pr[n] = teleport + d * sum + dangling_contribution;
        }

        // 归一化处理
        double total = 0.0;
        for (const auto &[k, v] : new_pr)
            total += v;
        for (auto &[k, v] : new_pr)
            v /= total;

        pr = new_pr;
    }

    return pr;
}

// 随机游走
void randomWalk(const Graph &g, const string &start)
{
    string current = start.empty() ? g.begin()->first : start;
    vector<string> path{current};
    unordered_set<string> visited_edges;

    while (true)
    {
        if (!g.count(current) || g.at(current).empty())
            break;

        // 按权重随机选择
        vector<pair<string, int>> edges(g.at(current).begin(), g.at(current).end());
        vector<int> weights;
        for (const auto &[_, w] : edges)
            weights.push_back(w);

        random_device rd;
        mt19937 gen(rd());
        discrete_distribution<> dist(weights.begin(), weights.end());
        string next = edges[dist(gen)].first;

        string edge = current + "->" + next;
        if (visited_edges.count(edge))
            break;
        visited_edges.insert(edge);

        path.push_back(next);
        current = next;

        cout << current << " ";
        this_thread::sleep_for(chrono::milliseconds(500)); // 控制速度
    }

    ofstream out("walk.txt");
    for (const auto &n : path)
        out << n << " ";
}

// 主函数集成
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }

    // 读取并构建图
    ifstream file(argv[1]);
    string text((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    auto [_, words] = clean_text(text);
    Graph g = build_graph(words);

    // 功能菜单
    while (true)
    {
        cout << "\n1. Show Graph\n2. Bridge Words\n3. New Text\n4. Shortest Path\n"
             << "5. PageRank\n6. Random Walk\nQ. Quit\n";

        char choice;
        cin >> choice;

        if (tolower(choice) == 'q')
            break;

        switch (choice)
        {
        case '1':
            showDirectedGraph(g, "graph");
            break;
        case '2':
        {
            string w1, w2;
            cin >> w1 >> w2;
            auto bridges = queryBridgeWords(g, w1, w2);
            cout << "Bridge words between " << w1 << " and " << w2 << ": ";
            for (const string &bridge : bridges)
                cout << bridge << " ";
            cout << endl;
            break;
        }
        case '3':
        {
            cin.ignore(numeric_limits<streamsize>::max(), '\n'); // 清除缓冲区
            cout << "请输入要插入桥接词的新文本: ";
            string userInput;
            getline(cin, userInput); // 读取整行输入
            string newText = generateNewText(userInput, g);
            cout << "生成的新文本: " << newText << endl;
            break;
        }
        case '4':
        {
            string start, end;
            cin >> start >> end;
            auto [path, distance] = shortestPath(g, start, end);
            cout << "Shortest path from " << start << " to " << end << ": ";
            for (const string &node : path)
                cout << node << " ";
            cout << "with distance: " << distance << endl;
            break;
        }
        case '5':
        {
            auto pr = pagerank(g);
            cout << "PageRank values:\n";
            for (const auto &[node, value] : pr)
            {
                cout << node << ": " << value << endl;
            }
            break;
        }
        case '6':
        {
            string start;
            cin >> start;
            cout << "Random Walk starting at " << start << ": ";
            randomWalk(g, start);
            cout << endl;
            break;
        }
        default:
            cout << "Invalid choice, please try again.\n";
        }
    }

    return 0;
}
