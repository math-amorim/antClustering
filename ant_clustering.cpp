#include <bits/stdc++.h>
#include <chrono>
#include <thread>
#include "debug.hpp"

using namespace std;

struct Item {
    bool exists = false;
    vector<double> features; // tamanho e peso
    int label = -1;
};

struct Ant {
    int x, y;
    bool carrying = false;
    Item carriedItem;
};

class AntClustering {
private:
    int rows, cols;
    int numItems;
    int numAnts;
    int radious = 2;
    double alpha =  5.0;
    double k1 = 0.1, k2 = 0.3;

    vector<vector<double>> centers = {
        {0.2, 0.2},
        {0.8, 0.2},
        {0.2, 0.8},
        {0.8, 0.8}
    };

    vector<vector<Item>> grid;
    vector<vector<bool>> antGrid;
    vector<Ant> ants;

    mt19937 rng;

public:
    AntClustering(int r, int c, int items, int antsCount) : rows(r), cols(c), numItems(items), numAnts(antsCount) {
        rng.seed(random_device{}());

        grid.assign(rows, vector<Item>(cols));
        antGrid.assign(rows, vector<bool>(cols, false));
        if (!loadItems()) {
            placeItems();
        }
        placeAnts();
    }

    void run(int steps, bool visualize = false, int delayMs = 80) {
        if (!visualize) {
            cout << "Estado inicial:\n";
            draw(0);
        }

        for (int t = 1; t <= steps; t++) {
            for (auto &ant : ants) {
                moveAnt(ant);
                act(ant);
            }

            if (visualize) {
                draw(t);
                this_thread::sleep_for(chrono::milliseconds(delayMs));
            }
        }

        if (!visualize) {
            cout << "Estado final:\n";
            draw(steps);
        }
    }

private:
    int wrap(int v, int lim) {
        if (v < 0) return lim-1;
        if (v >= lim) return 0;
        return v;
    }

    int randInt(int a, int b) {
        uniform_int_distribution<int> dist(a,b);
        return dist(rng);
    }

    double randDouble() {
        uniform_real_distribution<double> dist(0.0,1.0);
        return dist(rng);
    }

    void placeItems() {
        int placed = 0;
        while(placed < numItems) {
            int x = randInt(0, rows-1);
            int y = randInt(0, cols-1);

            if(!grid[x][y].exists) {
                int c = randInt(0, 3);
                double noise = 0.05;

                grid[x][y].exists = true;

                grid[x][y].features = { 
                    centers[c][0] + noise * randDouble(),
                    centers[c][1] + noise * randDouble()
                };

                grid[x][y].label = c;
                placed++;
            }
        }
    }

    bool loadItems() {
        vector<Item> items;
        string line;
        
        if (cin.rdbuf()->in_avail() == 0) {
            return false; 
        }

        while (getline(cin, line)) {
            line.erase(0, line.find_first_not_of(" \t"));
            if (line.empty() || line[0] == '#') continue;

            // troca vírgula por ponto
            for (char &c : line) {
                if (c == ',') c = '.';
            }

            stringstream ss(line);
            double x, y;
            int label;

            if (ss >> x >> y >> label) {
                Item it;
                it.exists = true;
                it.features = {x, y};
                it.label = label - 1;
                items.push_back(it);
            }
        }

        if (items.empty()) {
            return false; 
        }

        numItems = items.size();

        if (numItems == 0) {
            cerr << "ERRO: nenhum item carregado\n";
            exit(1);
        }

        int placed = 0;
        while (placed < numItems) {
            int gx = randInt(0, rows-1);
            int gy = randInt(0, cols-1);

            if (!grid[gx][gy].exists) {
                grid[gx][gy] = items[placed];
                placed++;
            }
        }
        return true;
    }

    void placeAnts() {
        int placed = 0;
        while(placed < numAnts) {
            int x = randInt(0, rows-1);
            int y = randInt(0, cols-1);

            if(!antGrid[x][y]) { 
                ants.push_back({x, y, false});
                antGrid[x][y] = true;
                placed++;
            }
        }
    }

    void moveAnt(Ant &ant) {
        static const vector<pair<int,int>> mvs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        auto [dx, dy] = mvs[randInt(0, 3)];

        int nx = wrap(ant.x + dx, rows);
        int ny = wrap(ant.y + dy, cols);

        if(!antGrid[nx][ny]) {
            antGrid[ant.x][ant.y] = false;
            ant.x = nx;
            ant.y = ny;
            antGrid[nx][ny] = true;
        }
    }

    int countItemsAround(int x, int y) {
        int cnt = 0;

        for(int dx = -radious; dx <= radious; dx++) {
            for(int dy = -radious; dy <= radious; dy++) {
                if(dx == 0 and dy == 0) continue;

                int nx = wrap(x + dx, rows);
                int ny = wrap(y + dy, cols);

                if(grid[nx][ny].exists) cnt++;
            }
        }

        return cnt;
    }

    double pickProbability(double f) {
        return pow(k1/(k1 + f), 2);
    }

    double dropProbability(double f) {
        return pow(f/(k2 + f), 2);
    }

    double euclidean(const vector<double>& a, const vector<double>& b) {
        double sum = 0;
        for (int i = 0; i < a.size(); i++) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return sqrt(sum);
    }

    double similarity(int x, int y, const Item& item) {
        double sum = 0;
        int count = 0;

        for (int dx = -radious; dx <= radious; dx++) {
            for (int dy = -radious; dy <= radious; dy++) {
                if (dx == 0 && dy == 0) continue;

                int nx = wrap(x + dx, rows);
                int ny = wrap(y + dy, cols);

                if (grid[nx][ny].exists) {
                    double d = euclidean(item.features, grid[nx][ny].features);

                    sum += max(0.0, 1 - (d / alpha));
                }
                count++;
            }
        }

        return sum / count;
    }

    void act(Ant &ant) {
        if(!ant.carrying) {
            if (grid[ant.x][ant.y].exists) {
                Item item = grid[ant.x][ant.y];
                double f = similarity(ant.x, ant.y, item);

                if(randDouble() < pickProbability(f)) {
                    ant.carriedItem = item;
                    grid[ant.x][ant.y].exists = false;
                    ant.carrying = true;
                }
            }
        }else {
            if(!grid[ant.x][ant.y].exists) {
                double f = similarity(ant.x, ant.y, ant.carriedItem);

                if(randDouble() < dropProbability(f)) {
                    grid[ant.x][ant.y] = ant.carriedItem;
                    ant.carrying = false;
                }
            }
        }
    }

void draw(int step, bool clear = false) {
    if (clear) {
        cout << "\033[2J\033[1;1H";
    }
        vector<string> view(rows, string(cols, ' '));

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if(grid[i][j].exists) view[i][j] = 'A' + grid[i][j].label;
            }
        }

        for(auto &ant : ants) view[ant.x][ant.y] = ant.carrying ? '*' : '.';

        string output;
        output += "Iteracao: " + to_string(step) + "\n\n";

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output += view[i][j];
                output += " ";
            }
            output += "\n";
        }
        cout << output;
        cout.flush();
    }
};

int main() {
    cin.tie(0)->sync_with_stdio(0);

    // Construtor recebe número de linhas, colunas, itens e formigas, respectivamente.
    AntClustering antC(64, 64, 400, 80);

    // Para rodar sem visualização de cada iteração, use: antC.run(numero_de_passos);
    antC.run(5000000);
    // Para rodar com visualização de cada iteração, use: antC.run(numero_de_passos, true, delay_em_milisegundos);
    // antC.run(200000, true, 50);

    return 0;
}


