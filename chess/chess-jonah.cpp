#include <bitset>
#include <format>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <map>

using uz = uint64_t;

static constexpr uz size = 2;

struct Board {
  using num_t = std::bitset<size>;
  using sol_t = std::map<uz, std::array<num_t, size>>;
  sol_t solutions;
  void solve() {

  }
  friend std::ostream &operator<<(std::ostream &os, Board &);
};

std::ostream &operator<<(std::ostream &os, Board &b) {
  os << "board:\n";
  for (auto it : b.solutions) {
    os << std::setw(4) << it.first << ": ";
    for (auto v : it.second)
      os << v << " ";
    os << "\n";
  }
  return os << "\n";
}

int main() {
  std::cout << "Board size: " << size << "\n";
  Board b{};
  b.solve();
  std::cout << "Solution: " << b << "\n";
  return 0;
}
