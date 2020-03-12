#include <vector>
#include <iostream>
#include <stdexcept>


class Heap
{
    public:
    explicit Heap(int capacity = 100): array(capacity), currentSize(0) {}
    explicit Heap(const std::vector<int> & item): array(item.size() + 10), currentSize(item.size())
    {
        for (int i = 0; i < item.size(); ++i)
            array[i + 1] = item[i];
        buildHeap();
    }
    ~Heap() = default;
    friend std::ostream &operator<<(std::ostream &, const Heap &);
    friend std::vector<int> heapSort(Heap &);

    bool isEmpty() const;
    const int & findTop() const;

    void insert(const int &x);
    void deleteTop();
    int popTop();
    // void deleteTop( int & minItem);
    // void makeEmpty();
private:
    int currentSize;
    std::vector<int> array;
    void buildHeap();
    void percolateDown(int hole);

};

std::vector<int> heapSort(Heap &h)
{
    int originalSize = h.currentSize;
    std::vector<int> res(originalSize);
    for (int i = 0; i < originalSize; ++i)
    {
        res[i] = h.popTop();
    }
    return res;
}

std::ostream &operator<<(std::ostream &os, const Heap &h)
{
    for (int i = 1; i <= h.currentSize; ++i)
    {
        os << h.array[i] << ",";
    }
    os << "\b ";
    return os;
}

bool Heap::isEmpty() const
{
    if (currentSize == 0)
        return true;
    return false;
}

const int & Heap::findTop() const
{
    if (isEmpty())
        throw std::underflow_error("oooh!");
    return array[1];
}

void Heap::insert(const int &x)
{
    ++currentSize;
    if (currentSize > array.size() - 1)
        array.resize(array.size() * 2);
    int pos = currentSize;
    for (; pos > 1 && array[pos / 2] < x; pos /= 2)
    {
        array[pos] = array[pos / 2];
    }
    array[pos] = x;
}

void Heap::deleteTop()
{
    if (isEmpty())
        throw std::underflow_error("oooh!");
    array[1] = array[currentSize--];
    percolateDown(1);

}

int Heap::popTop()
{
    if (isEmpty())
        throw std::underflow_error("oooh!");
    int res = array[1];
    array[1] = array[currentSize--];
    percolateDown(1);

    return res;
}

void Heap::buildHeap()
{
    for (int i = currentSize / 2; i > 0; --i)
        percolateDown(i);
}

void Heap::percolateDown(int hole)
{
    int child;
    auto e = array[hole];
    for (; (hole << 1) <= currentSize && e < array[hole << 1]; hole=child)
    {
        child = hole << 1;
        if (child + 1 <= currentSize && array[child] < array[child + 1])
            ++child;
        array[hole] = array[child];
    }
    array[hole] = e;
}





int main()
{
    std::vector<int> v1 = {5, 2, 6, 9, 10, 1, 3};
    for (auto e : v1)
        std::cout << e << std::endl;

    Heap h = Heap(5);
    std::cout << h.isEmpty() << std::endl;

    for (int i = 0; i < 10; ++i)
        h.insert(i);
    std::cout << h << std::endl;
    h.deleteTop();
    std::cout << h << std::endl;

    Heap h2 = Heap(v1);
    std::cout << h2 << std::endl;

    std::cout << h2.isEmpty() << std::endl;
    std::cout << h2.findTop() << std::endl;

    for (int i = 0; i < v1.size(); ++i)
    {
        h2.deleteTop();
        std::cout << h2 << std::endl;

    }

    auto res = heapSort(h2);
    for (auto e : res)
        std::cout << e << ", ";

}